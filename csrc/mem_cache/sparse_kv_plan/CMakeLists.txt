sgl_register_operator(
    ID sparse_kv_plan
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseKvPlan
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/sparse_kv_plan.cpp
    INFER_SOURCES op_host/sparse_kv_plan.cpp
    TILING_SOURCES op_host/sparse_kv_plan.cpp
    KERNEL_SOURCES op_kernel/sgl_sparse_kv_plan.cpp
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
)
