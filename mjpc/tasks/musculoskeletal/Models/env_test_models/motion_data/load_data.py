'''
pelvis tz 0

pelvis ty 1

pelvis tx 2

pelvis tilt 3

pelvis list 4

pelvis rotation 5



sternoclavicular_r2_r 69

sternoclavicular_r3_r 70

unrotscap_r3_r 71

unrotscap_r2_r 72

acromioclavicular_r2_r 73

acromioclavicular_r3_r 74

acromioclavicular_r1_r 75

unrothum_r1_r 76

unrothum_r3_r 77

unrothum_r2_r 78

elv_angle_r 79

shoulder_elv_r 80

shoulder1_r2_r 81

shoulder_rot_r 82

elbow_flexion_r 83

pro_sup_r 84

deviation_r 85

flexion_r 86

wrist_hand_r1_r 87

wrist_hand_r3_r 88



left + 20

89 = 69 + 20

where is the equality? -- check the mot



hip_flexion_r 178

hip_flexion_r 6 + 172

hip_adduction_r 7

hip_rotation_r 8

knee_angle_r_translation2 9

knee_angle_r_translation1 10

knee_angle_r 11

knee_angle_r_rotation2 12

knee_angle_r_rotation3 13

ankle_angle_r 14

subtalar_angle_r 15

mtp_angle_r 16

knee_angle_r_beta_translation2 17

knee_angle_r_beta_translation1 18

knee_angle_r_beta_rotation1 19



hip_flexion_l 20

hip_adduction_l 21

hip_rotation_l 22



knee_angle_l_translation2 23

knee_angle_l_translation1 24

knee_angle_l 25

knee_angle_l_rotation2 26

knee_angle_l_rotation3 27

ankle_angle_l 28

subtalar_angle_l 29

mtp_angle_l 30

knee_angle_l_beta_translation2 31

knee_angle_l_beta_translation1 32

knee_angle_l_beta_rotation1 33 + 172 = 205

'''
import numpy as np
import pandas as pd

def load_data(file_path="./rajagopal2015_walking.csv"):
    df = pd.read_csv(file_path, sep="\t", header=6)

    data_time = df["time"]

    # load data
    mj_joint_data = np.zeros((len(data_time), 206))
    mj_joint_data[:, 2] = np.array(df["pelvis_tz"])
    mj_joint_data[:, 1] = np.array(df["pelvis_ty"])
    mj_joint_data[:, 0] = np.array(df["pelvis_tx"])

    mj_joint_data[:, 3] = np.array(df["pelvis_tilt"])
    mj_joint_data[:, 4] = np.array(df["pelvis_list"])
    mj_joint_data[:, 5] = np.array(df["pelvis_rotation"])

    mj_joint_data[:, 6+172] = np.array(df["hip_flexion_r"])
    mj_joint_data[:, 7+172] = np.array(df["hip_adduction_r"])
    mj_joint_data[:, 8+172] = np.array(df["hip_rotation_r"])

    mj_joint_data[:, 11+172] = np.array(df["knee_angle_r"])

    mj_joint_data[:, 14+172] = np.array(df["ankle_angle_r"])
    mj_joint_data[:, 15+172] = np.array(df["subtalar_angle_r"])
    mj_joint_data[:, 16+172] = np.array(df["mtp_angle_r"])

    mj_joint_data[:, 20+172]=np.array(df["hip_flexion_l"])
    mj_joint_data[:, 21+172]=np.array(df["hip_adduction_l"])
    mj_joint_data[:, 22+172]=np.array(df["hip_rotation_l"])

    mj_joint_data[:, 25+172]=np.array(df["knee_angle_l"])

    mj_joint_data[:, 28+172]=np.array(df["ankle_angle_l"])
    mj_joint_data[:, 29+172]=np.array(df["subtalar_angle_l"])
    mj_joint_data[:, 30+172]=np.array(df["mtp_angle_l"])

    

    # degree to rad
    mj_joint_data = mj_joint_data / 180 * np.pi

    # mask_joint = [0] * 34

    # for i in [9, 10, 12, 13, 17, 18, 19, 23, 24, 26, 27, 31, 32, 33]:
    #     mask_joint[i] = 1
    
    total_set = set([i for i in range(34)])
    mask_joint = set([9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 26, 27, 29, 30, 31, 32, 33])
    
    mask_joint = list(total_set - mask_joint)

    return mj_joint_data, mask_joint, data_time.tolist()
