from models.conformer import (
    ConformerLayer,
    SqueezeExcitation,
)


class FastConformerLayer(ConformerLayer):
    """Conformer block with optional squeeze-excitation on the conv branch."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_module_kernel: int = 9,
        dropout: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            cnn_module_kernel=cnn_module_kernel,
            dropout=dropout,
            use_se=use_se,
        )
