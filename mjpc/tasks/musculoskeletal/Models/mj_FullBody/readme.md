# FullBody Model

MuJoCo Full Body Model

- Lower Limbs with wrapper geoms
- Upper Limbs with wrapper geoms
- Simple Hands and Feet
- Simple Spine

## Action Space

- muscles : 700
    - lowerbody: 39 * 2
    - upperbody: 47 * 2
    - torso: 264 * 2
    - 降维后torso: 15 * 2
    - 降维后full: 202



- torso降维方案
  - 可视化检查之后，按文件中的肌群降维
    - Psoas Major 腰大肌, 1--11, 11
    - Rectus Abdominus 腹直肌. 12, 1
    - Sacro Spinalis 竖脊肌, 13--50, 38
    - Quadratus Laborum 腰方肌, 51--68, 18
    - Transverso Spinalis 横突棘肌, 69--93（腰部）, 25
    - External Oblique 腹外斜肌, 94--101, 8
    - Internal Oblique 腹内斜肌, 102--107, 6
    - Latissimus Dorsi 背阔肌, 108--121, 14
    - Neck 脖颈肌, 122-129（向前拉，低头部分）, 8
    - Trapezius 斜方肌, 130--143, 14
    - Neck 脖颈肌，144--154（向后拉，仰头部分）, 11
    - Transverso Spinalis 横突棘肌,  155--173（腰部以上）, 19
    - Serratus Anterior 前锯肌, 174--183, 10
    - Transversus Abdominus 腹横肌, 184--188, 5
    - EXT_IS and INT_IS 肋间肌，所有都降成一维, 189--264, 76

## Observation Space

- joints: 85
    - pelvis: 6
    - lowerbody: 30
    - torso: 9
    - upperbody: 40

