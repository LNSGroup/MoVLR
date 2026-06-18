# FullBody Model

MuJoCo Full Body Model


## Joint List

- Total Number: 251
  - Pelvis: 6

  - Leg_r: 9
  - Foot_r: 26
  - Patella_r: 3

  - Leg_l: 9
  - Foot_l: 26
  - Patella_l: 3

  - Abdomen: 3
  - Lumbar: 15
  - Thoracic: 36
  - Cervical: 24
  - Head: 7

  - Arm_r: 16
  - Hand_r: 26

  - Arm_l: 16
  - Hand_l: 26



[0:6]
'pelvis_tz', 'pelvis_ty', 'pelvis_tx', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 
[6:15]
'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'walker_knee_r_translation1', 'walker_knee_r_translation2', 'walker_knee_r_translation3', 'knee_angle_r', 'walker_knee_r_rotation2', 'walker_knee_r_rotation3', 
[15:41]
'ankle_flexion_r', 'ankle_in_ev_r', 'ankle_rot_r', 'subtalar_angle_r', 'midtarsal_angle_r', 'tarsometatarsal_angle_1_r', 'mtp_angle_1_r', 'toe1_lateral_r', 'itp_1_r', 'tarsometatarsal_angle_2_r', 'mtp_angle_2_r', 'itp_prox_2_r', 'itp_dist_2_r', 'tarsometatarsal_angle_3_r', 'mtp_angle_3_r', 'itp_prox_3_r', 'itp_dist_3_r', 'tarsometatarsal_angle_4_r', 'mtp_angle_4_r', 'itp_prox_4_r', 'itp_dist_4_r', 'tarsometatarsal_angle_5_r', 'mtp_angle_5_r', 'toe5_lateral_r', 'itp_prox_5_r', 'itp_dist_5_r', 
[41:44]
'patellofemoral_r_translation1', 'patellofemoral_r_translation2', 'patellofemoral_r_rotation1', 
[44:53]
'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'walker_knee_l_translation1', 'walker_knee_l_translation2', 'walker_knee_l_translation3', 'knee_angle_l', 'walker_knee_l_rotation2', 'walker_knee_l_rotation3', 
[53:79]
'ankle_flexion_l', 'ankle_in_ev_l', 'ankle_rot_l', 'subtalar_angle_l', 'midtarsal_angle_l', 'tarsometatarsal_angle_1_l', 'mtp_angle_1_l', 'toe1_lateral_l', 'itp_1_l', 'tarsometatarsal_angle_2_l', 'mtp_angle_2_l', 'itp_prox_2_l', 'itp_dist_2_l', 'tarsometatarsal_angle_3_l', 'mtp_angle_3_l', 'itp_prox_3_l', 'itp_dist_3_l', 'tarsometatarsal_angle_4_l', 'mtp_angle_4_l', 'itp_prox_4_l', 'itp_dist_4_l', 'tarsometatarsal_angle_5_l', 'mtp_angle_5_l', 'toe5_lateral_l', 'itp_prox_5_l', 'itp_dist_5_l', 
[79:82]
'patellofemoral_l_translation1', 'patellofemoral_l_translation2', 'patellofemoral_l_rotation1', 
[82:85]
'Abs_FE', 'Abs_LB', 'Abs_AR', 
[85:100]
'lumbar_FE', 'lumbar_LB', 'lumbar_AR', 'L4_L5_IVDjnt_r3', 'L4_L5_IVD_bendjnt_r1', 'L4_L5_IVD_rotjnt_r2', 'L3_L4_IVDjnt_r3', 'L3_L4_IVD_bendjnt_r1', 'L3_L4_IVD_rotjnt_r2', 'L2_L3_IVDjnt_r3', 'L2_L3_IVD_bendjnt_r1', 'L2_L3_IVD_rotjnt_r2', 'L1_L2_IVDjnt_r3', 'L1_L2_IVD_bendjnt_r1', 'L1_L2_IVD_rotjnt_r2', 
[100:136]
'thoracic_FE', 'thoracic_LB', 'thoracic_AR', 'T11_T12_FE', 'T11_T12_LB', 'T11_T12_AR', 'T10_T11_FE', 'T10_T11_LB', 'T10_T11_AR', 'T9_T10_FE', 'T9_T10_LB', 'T9_T10_AR', 'T8_T9_FE', 'T8_T9_LB', 'T8_T9_AR', 'T7_T8_FE', 'T7_T8_LB', 'T7_T8_AR', 'T6_T7_FE', 'T6_T7_LB', 'T6_T7_AR', 'T5_T6_FE', 'T5_T6_LB', 'T5_T6_AR', 'T4_T5_FE', 'T4_T5_LB', 'T4_T5_AR', 'T3_T4_FE', 'T3_T4_LB', 'T3_T4_AR', 'T2_T3_FE', 'T2_T3_LB', 'T2_T3_AR', 'T1_T2_FE', 'T1_T2_LB', 'T1_T2_AR', 
[136:160]
'cervical_FE', 'cervical_LB', 'cervical_AR', 'aux7jnt_r3', 'aux7jnt_r1', 'aux7jnt_r2', 'aux6jnt_r3', 'aux6jnt_r1', 'aux6jnt_r2', 'aux5jnt_r3', 'aux5jnt_r1', 'aux5jnt_r2', 'aux4jnt_r3', 'aux4jnt_r1', 'aux4jnt_r2', 'aux3jnt_r3', 'aux3jnt_r1', 'aux3jnt_r2', 'head_FE', 'head_LB', 'head_AR', 'aux1jnt_r3', 'aux1jnt_r1', 'aux1jnt_r2', 
[160:167]
'jawjnt', 'eye_add_abd_r', 'eye_ele_dep_r', 'eye_inc_exc_r', 'eye_add_abd_l', 'eye_ele_dep_l', 'eye_inc_exc_l', 
[167:183]
'sternoclavicular_r2_r', 'sternoclavicular_r3_r', 'unrotscap_r3_r', 'unrotscap_r2_r', 'acromioclavicular_r2_r', 'acromioclavicular_r3_r', 'acromioclavicular_r1_r', 'unrothum_r1_r', 'unrothum_r3_r', 'unrothum_r2_r', 'elv_angle_r', 'shoulder_elv_r', 'shoulder1_r2_r', 'shoulder_rot_r', 'elbow_flexion_r', 'pro_sup_r', 
[183:209]
'deviation_r', 'flexion_r', 'wrist_hand_r_r1_r', 'wrist_hand_r_r3_r', 'cmc_flexion_r', 'cmc_abduction_r', 'mp_flexion_r', 'ip_flexion_r', '2mcp_abduction_r', '2mcp_flexion_r', '2pm_flexion_r', '2md_flexion_r', '3mcp_abduction_r', '3mcp_flexion_r', '3pm_flexion_r', '3md_flexion_r', '4cmc_flexion_r', '4mcp_abduction_r', '4mcp_flexion_r', '4pm_flexion_r', '4md_flexion_r', 'CMC5_r_r1_r', '5mcp_abduction_r', '5mcp_flexion_r', '5pm_flexion_r', '5md_flexion_r', 
[209:225]
'sternoclavicular_r2_l', 'sternoclavicular_r3_l', 'unrotscap_r3_l', 'unrotscap_r2_l', 'acromioclavicular_r2_l', 'acromioclavicular_r3_l', 'acromioclavicular_r1_l', 'unrothum_r1_l', 'unrothum_r3_l', 'unrothum_r2_l', 'elv_angle_l', 'shoulder_elv_l', 'shoulder1_r2_l', 'shoulder_rot_l', 'elbow_flexion_l', 'pro_sup_l', 
[225:251]
'deviation_l', 'flexion_l', 'wrist_hand_r_r1_l', 'wrist_hand_r_r3_l', 'cmc_flexion_l', 'cmc_abduction_l', 'mp_flexion_l', 'ip_flexion_l', '2mcp_abduction_l', '2mcp_flexion_l', '2pm_flexion_l', '2md_flexion_l', '3mcp_abduction_l', '3mcp_flexion_l', '3pm_flexion_l', '3md_flexion_l', '4cmc_flexion_l', '4mcp_abduction_l', '4mcp_flexion_l', '4pm_flexion_l', '4md_flexion_l', 'CMC5_r_r1_l', '5mcp_abduction_l', '5mcp_flexion_l', '5pm_flexion_l', '5md_flexion_l'


## Muscle List

- Total Number: 1262
  - Right: 631
    - Leg: 56
    - Foot: 40
    - Arm: 24
    - Hand: 44
    - Back: 259
    - Thorax_Abdomen: 129
    - Neck: 33
    - Head: 46
  - Left: 631

