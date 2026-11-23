#!/usr/bin/env bash
# Make and build the CUTLASS sm_120 block-scaled GEMM variants the gap analysis times:
# 79a (NVFP4 x NVFP4 -> BF16), 79c changed to MXFP8 x MXFP8 -> BF16, 79d (grouped NVFP4)
# and 79d changed to grouped MXFP8. The tile shape comes from TBM, TBN and TBK.
# usage: cutlass_variants.sh <cutlass-dir> <outdir> <tbm> <tbn> <tbk>
set -euo pipefail
cutlass=$1 out=$2 m=$3 n=$4 k=$5
ex=$cutlass/examples/79_blackwell_geforce_gemm
mkdir -p "$out/src"
tile="s/Shape<_128,_128,_128>/Shape<_${m},_${n},_${k}>/"
sed "$tile" "$ex/79a_blackwell_geforce_nvfp4_bf16_gemm.cu" > "$out/src/a_nvfp4.cu"
sed -e "$tile" -e 's/mx_float6_t<cutlass::float_e3m2_t>/mx_float8_t<cutlass::float_e4m3_t>/' \
    -e 's/AlignmentB  = 128;/AlignmentB  = 16;/' \
    "$ex/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu" > "$out/src/c_mxfp8.cu"
sed "$tile" "$ex/79d_blackwell_geforce_nvfp4_grouped_gemm.cu" > "$out/src/d_nvfp4.cu"
sed -e "$tile" -e 's/ElementInput = cutlass::float_e2m1_t/ElementInput = cutlass::float_e4m3_t/' \
    -e 's/nv_float4_t<ElementInput>/mx_float8_t<ElementInput>/' \
    -e 's/AlignmentA  = 32;/AlignmentA  = 16;/' -e 's/AlignmentB  = 32;/AlignmentB  = 16;/' \
    -e 's/std::vector<HostTensorSF> block_SFD;/std::vector<HostTensorSFD> block_SFD;/' \
    -e 's/std::vector<HostTensorSF> block_ref_SFD;/std::vector<HostTensorSFD> block_ref_SFD;/' \
    -e 's/block_SFD.push_back(HostTensorSF(/block_SFD.push_back(HostTensorSFD(/' \
    -e 's/block_ref_SFD.push_back(HostTensorSF(/block_ref_SFD.push_back(HostTensorSFD(/' \
    -e 's/DeviceAllocation<typename Gemm::GemmKernel::CollectiveMainloop::ElementSF \*> ptr_SFD;/DeviceAllocation<ElementSFD *> ptr_SFD;/' \
    -e 's/std::vector<typename Gemm::GemmKernel::CollectiveMainloop::ElementSF \*> ptr_SFD_host/std::vector<ElementSFD *> ptr_SFD_host/' \
    -e 's/^using HostTensorSF = \(.*\)$/using HostTensorSF = \1\nusing HostTensorSFD = cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout>;/' \
    "$ex/79d_blackwell_geforce_nvfp4_grouped_gemm.cu" > "$out/src/d_mxfp8.cu"
for v in ${VARIANTS:-a_nvfp4 c_mxfp8 d_nvfp4 d_mxfp8}; do
    bash "$(dirname "$0")/build_cutlass.sh" "$cutlass" "$out/src/$v.cu" "$out/${v}_${m}x${n}x${k}" \
        -diag-suppress 20012 2>&1 | grep -E "error" | head -5 || true
done
ls -la "$out"
