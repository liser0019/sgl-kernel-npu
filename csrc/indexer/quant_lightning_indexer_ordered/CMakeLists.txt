sgl_register_operator(
    ID quant_lightning_indexer_ordered
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglQuantLightningIndexerOrdered
    TORCH_ADAPTER torch_adapter.cpp
    SUPPORTED_ARCHS 3510
    OPDEF_SOURCES op_host/quant_lightning_indexer_ordered_def.cpp
    INFER_SOURCES op_host/quant_lightning_indexer_ordered_infershape.cpp
    TILING_SOURCES op_host/quant_lightning_indexer_ordered_tiling.cpp
    KERNEL_SOURCES op_kernel/sgl_quant_lightning_indexer_ordered.cpp
    KERNEL_COMPILE_OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
)
