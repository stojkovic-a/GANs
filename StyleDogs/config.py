image_dir='./Real/R1/Edited'
start_train_img_size=4
lr_gen=0.001
# lr_gen=0.0001
lr_disc=0.001
# lr_disc=0.0001
# batch_sizes=[256,256,128,64,32,16  ]
batch_sizes=[32,32, 16,8,4,2  ]
factors=[1,1,1,1/2,1/4,1/8,1/16,1/32]
num_channels=3
z_depth=512
w_depth=512
in_channels=512
lambda_gp=10
# progressive_epoches=[30]*len(batch_sizes)
progressive_epoches=[40,50,60,70,80,90]
allowed_format=('.jpg','.png','.tif')
beta1=0.0
beta2=0.99
examples_save_dir='training_results'
examples_number=50
cycle_save=12
generators_save_dir='./Gens'
discriminators_save_dir='./Discs'
image_size=256
binary_image_save_dir='examples_binary'
binary_image_generation_dir='generated_binary'


discriminator_epoches=80
generator_epoches=20

# num_cycles=50000
# batch_size=25
# num_classes=20

# starting_layer_num_channels=1024

# dataset_annotations='./Real/R1/Annotation'
# save_dir='./Real/R1/Edited128'
# num_classes=120
