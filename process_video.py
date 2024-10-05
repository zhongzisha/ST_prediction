

import sys,os,glob,cv2

root = '/Users/zhongz2/Desktop/aws/tests'
files = glob.glob(root+'/*.mov')
lines = []
for f in files:
    dst = os.path.join(root, '..', os.path.basename(f).replace('.mov', '.mp4'))
    if os.path.exists(dst):
        print(f)
        continue
    vid = cv2.VideoCapture(f)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(height, width)

    lines.append('ffmpeg -i "{}" -vf "crop=1600:1964:712:0" "{}"\nsleep(1);\n'.format(f, dst))

with open('do.sh', 'w') as fp:
    fp.writelines(lines)







