dataset = "coco"

teacher_pretrained = True

img_size = 224
batch_size = 64

model_t = "resnet101"
lr_t = 1e-4
max_epoch_t = 80
stop_epoch_t = 30

model_s = "resnet34"
lr_s = 1e-4
max_epoch_s = 80
stop_epoch_s = 80

criterion_t2s_para = dict(
    name="L2D",
    para=dict(
        lambda_ft=0.0,
        ft_dis=None,
        lambda_le=1.0,
        le_dis=dict(
            name="LED",
            para=dict(
                lambda_cd=100.0,
                lambda_id=1000.0
            )
        ),
        lambda_logits=10.0,
        logits_dis=dict(
            name="MLD",
            para=dict()
        )
    )
)
