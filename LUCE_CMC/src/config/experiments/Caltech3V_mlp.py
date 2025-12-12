from config.defaults import Experiment, DDC, Fusion, MLP, Loss, Dataset, LUCECMC, Optimizer

caltech3v = Experiment(
    dataset_config=Dataset(name="Caltech-3V"),
    model_config=LUCECMC(
        backbone_configs=(
            MLP(input_size=(40,)),
            MLP(input_size=(254,)),
            MLP(input_size=(928,)),
        ),
        fusion_config=Fusion(method="adaptive_fusion", n_views=3),
        projector_config=None,
        cm_config=DDC(n_clusters=7),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|EnFeaCL|EnCluCL",
            delta=20.0
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.1)
    ),

)
