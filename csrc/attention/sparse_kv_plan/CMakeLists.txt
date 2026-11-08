sgl_register_operator(
    ID sparse_kv_plan
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglSparseKvPlan
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/sparse_kv_plan_def.cpp
    INFER_SOURCES op_host/sparse_kv_plan_infershape.cpp
    TILING_SOURCES op_host/sparse_kv_plan_tiling.cpp
    TORCH_ADAPTER torch_adapter.cpp
)
sgl_register_operator_variant(
    ID sparse_kv_plan ARCH 3510
    KERNEL_SOURCES op_kernel/sgl_sparse_kv_plan_apt.cpp
    KERNEL_COMPILE_OPTIONS
        "--cce-auto-sync=off"
        "-Wno-deprecated-declarations"
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
)
