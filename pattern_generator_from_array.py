import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint

outfile = "/Users/anzalks/Documents/git_sync/ep_snippets/generated_patterns"

grid_size = [24,24]

point_0 = np.zeros(grid_size)
point_0[11,0]=225
point_1 = np.zeros(grid_size)
point_1[11,2]=225
point_2 = np.zeros(grid_size)
point_2[11,4]=225
point_3 = np.zeros(grid_size)
point_3[11,6]=225
point_4 = np.zeros(grid_size)
point_4[11,8]=225
point_5 = np.zeros(grid_size)
point_5[11,10]=225
point_6 = np.zeros(grid_size)
point_6[11,12]=225
point_7 = np.zeros(grid_size)
point_7[11,14]=225
point_8 = np.zeros(grid_size)
point_8[11,16]=225
point_9 = np.zeros(grid_size)
point_9[11,18]=225
point_10 = np.zeros(grid_size)
point_10[11,20]=225
point_11 = np.zeros(grid_size)
point_11[11,22]=225


points = [point_0,point_1,point_2,point_3,point_4,point_5,point_6,
          point_7,point_8,point_9,point_10,point_11]
for i,im in enumerate(points):
    im.astype(np.uint8)
    img = Image.fromarray(im)
    img= img.convert('RGB')
    img.save(f'{outfile}/point_{i+1}.png')
pat_1 = points[0]+points[1]+points[2]+points[3]+points[4]
pat_2 = points[2]+points[3]+points[4]+points[5]+points[6]
pat_3 = points[3]+points[4]+points[5]+points[6]+points[7]
pat_4 = points[4]+points[5]+points[6]+points[7]+points[8]
pat_5 = points[5]+points[6]+points[7]+points[8]+points[9]
pat_6 = points[6]+points[7]+points[8]+points[9]+points[10]
pat_7 = points[7]+points[8]+points[9]+points[10]+points[11]

patterns = [pat_1,pat_2,pat_3,pat_4,pat_5,pat_6,pat_7]
for i,im in enumerate(patterns):
    im.astype(np.uint8)
    img = Image.fromarray(im)
    img= img.convert('RGB')
    img.save(f'{outfile}/pattern_{i+1}.png')
