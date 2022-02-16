# sh experiments/tinyimnet-twentytask.sh n path/to/imgnet

# process inputs
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}
DEFAULTDR="datasets"
DATAROOT=${2:-$DEFAULTDR}

# benchmark settings
DATE=ICCV2021
SPLIT=10
OUTDIR=outputs/${DATE}/DFCIL-fivetask/ImageNet-50

###############################################################

# make save directory
mkdir -p $OUTDIR

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=5

# hard coded inputs
REPEAT=1
SCHEDULE="30 60 80 90 100"
PI=50000
MODELNAME=resnet32
BS=32
WD=0.0001
MOM=0.9
OPT="SGD"
LR=0.1
 
# #########################
# #         OURS          #
# #########################

# # Full Method
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu 1e-1 --memory 0 --model_name $MODELNAME --model_type resnet \
#     --learner_type datafree --learner_name AlwaysBeDreaming \
#     --gen_model_name IMNET_GEN --gen_model_type generator \
#     --beta 1 --power_iters $PI --deep_inv_params 1e-3 5e1 1e-3 1e3 1 \
#     --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/abd

# #########################
# #  BASELINES  EXISTING  #
# #########################

# # Oracle
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0 --model_name $MODELNAME --model_type resnet \
#     --learner_type default --learner_name NormalNN --oracle_flag \
#     --overwrite 0 --max_task $MAXTASK --log_dir ${OUTDIR}/oracle

# # Base
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 0 --model_name $MODELNAME --model_type resnet \
#     --learner_type default --learner_name NormalNN \
#     --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/base

# # LwF
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu 1 --memory 0 --model_name $MODELNAME --model_type resnet \
#     --learner_type kd --learner_name LWF \
#     --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/lwf

# # LwF.MC
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --mu 1 --memory 0 --model_name $MODELNAME --model_type resnet \
#     --learner_type kd --learner_name LWF_MC \
#     --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_mc

# # Naive Rehearsal
# python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
#     --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#     --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#     --memory 2000 --model_name $MODELNAME --model_type resnet \
#     --learner_type default --learner_name NormalNN \
#     --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/rehearsal

# LwF - Coreset
python -u run_dfcil.py --dataset ImageNet --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT --dataroot $DATAROOT \
    --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
    --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
    --mu 1 --memory 2000 --model_name $MODELNAME --model_type resnet \
    --learner_type kd --learner_name LWF \
    --overwrite $OVERWRITE --max_task $MAXTASK --log_dir ${OUTDIR}/lwf_coreset