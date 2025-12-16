import matplotlib.pyplot as plt
from PIL import Image
import os


def make():
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

data_type = 'fix'
container_type = 'multi'
pack_type = 'last'

pack_ideal = "pack_ideal"
# path = f"./{pack_ideal}/{data_type}/{container_type}/{pack_type}"

old_folders = os.listdir(f"./{pack_ideal}/{data_type}")

required = [
    105, 107, 110, 121, 124, 126, 138, 141, 144, 146, 147, 150, 152, 155, 157, 158, 159, 169
]

folders = []
for f in old_folders:
    if os.path.exists( os.path.join(f"./{pack_ideal}/{data_type}/{f}/{container_type}/greedy_last-0.png") ) and os.path.exists( os.path.join(f"./{pack_ideal}/{data_type}/{f}/{container_type}/greedy_all-0.png") ):
        # if f in [ str(i) for i in required ]:
        if True:
            folders.append(f)
        

num = len(folders)
print(num)



ctn_each = 5
if container_type == 'multi':
    col_num = 2 + 3 * ctn_each
else:
    col_num = 5

plt.figure(figsize=(col_num, num*2))

count = 1
for i in range(num):
    for pack_type in ['all', 'last']:
        plt.subplot( num*2, col_num, count )
        count+= 1
        plt.title(f"{folders[i]}-{pack_type}")
        make()

        img_name = f"./{pack_ideal}/{data_type}/{folders[i]}/init-0.png"
        try:
            img = Image.open(img_name)
        except:
            img = None 

        plt.subplot( num*2, col_num, count )
        if img is not None:
            plt.imshow(img)
        make()

        
        count+=1

        for j, method in enumerate(['tn', 'greedy', 'tnpp']):

            if container_type == 'multi':
                max_ctn = ctn_each
            else:
                max_ctn = 1

            for ctn_i in range(max_ctn):
                plt.subplot( num*2, col_num, count )
                try:
                    img = Image.open(f"./{pack_ideal}/{data_type}/{folders[i]}/{container_type}/{method}_{pack_type}-{ctn_i}.png")
                except:
                    img = None 

                if img is not None:
                    plt.imshow(img)
                make()

                count+=1

# plt.savefig(f"./{pack_ideal}/{data_type}-{container_type}-{pack_type}.pdf", bbox_inches='tight')
plt.savefig(f"./{pack_ideal}/{data_type}-{container_type}-new3.pdf", bbox_inches='tight')



