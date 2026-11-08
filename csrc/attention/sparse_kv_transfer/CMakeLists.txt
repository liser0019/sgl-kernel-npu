sgl_register_operator(
    ID sparse_kv_transfer
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseKvTransfer
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/sparse_kv_transfer_def.cpp
    INFER_SOURCES op_host/sparse_kv_transfer_infershape.cpp
    TILING_SOURCES op_host/sparse_kv_transfer_tiling.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID sparse_kv_transfer ARCH 3510
    KERNEL_SOURCES op_kernel/sgl_sparse_kv_transfer_apt.cpp
    KERNEL_COMPILE_OPTIONS
        "--cce-auto-sync=off"
        "-Wno-deprecated-declarations"
)
