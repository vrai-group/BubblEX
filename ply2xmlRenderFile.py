import numpy as np
from plyfile import PlyData

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.5.0">
	<integrator type="path"/>

	<sensor type="perspective">
		<float name="farClip" value="100"/>
		<float name="focusDistance" value="7.3931"/>
		<float name="fov" value="18.8805"/>
		<string name="fovAxis" value="x"/>
		<float name="nearClip" value="0.1"/>
		<transform name="toWorld">

			<lookat target="3.2061, 3.26478, 3.15921" origin="3.78345, 3.84213, 3.73656" up="-0.408248, -0.408248, 0.816497"/>
		</transform>

		<sampler type="ldsampler">
			<integer name="sampleCount" value="256"/>
		</sampler>

		<film type="ldrfilm">
			<boolean name="banner" value="false"/>
			<float name="exposure" value="0"/>
			<float name="gamma" value="-1"/>
			<integer name="height" value="512"/>
			<string name="tonemapMethod" value="gamma"/>
			<integer name="width" value="512"/>

			<rfilter type="gaussian"/>
		</film>
	</sensor>

	<bsdf type="roughplastic" id="surfaceMaterial">
		<string name="distribution" value="ggx"/>
		<float name="alpha" value="0.05"/>
		<float name="intIOR" value="1.46"/>
		<rgb name="diffuseReflectance" value="1,1,1"/>
		<!-- default 0.5 -->
	</bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
	<shape type="rectangle">
		<ref name="bsdf" id="surfaceMaterial"/>
		<transform name="toWorld">
			<scale x="10" y="10" z="1"/>
			<translate x="0" y="0" z="-1"/>
		</transform>
	</shape>

	<shape type="rectangle">
		<transform name="toWorld">
			<scale x="10" y="10" z="1"/>

			<lookat target="0,0,0" origin="-4,4,20" up="0,0,1"/>
		</transform>

		<emitter type="area">
			<rgb name="radiance" value="6,6,6"/>
		</emitter>
	</shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]
xml_segments = [xml_head]

def read_color_ply(filename):
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    r = np.asarray(plydata.elements[0].data['red'])
    g = np.asarray(plydata.elements[0].data['green'])
    b = np.asarray(plydata.elements[0].data['blue'])
    return np.stack([x,y,z,r,g,b], axis=1)

# MAIN

#filename="results/activationPlot/a_677_gt32_p32.ply"
filename="results/gradcamPlot/ag_median_677_tg32_gt32_p32.ply"

pcl= read_color_ply(filename)
cols=pcl[:,3:]/255

#pcl = np.load('chair_pcl.npy')
# pcl = standardize_bbox(pcl, 2048)
# pcl = pcl[:,[2,0,1]]
# pcl[:,0] *= -1
# pcl[:,2] += 0.0125

for i in range(pcl.shape[0]):
    #color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
    color = cols[i]
    xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open(filename.split(".")[0] + '.xml', 'w') as f:
    f.write(xml_content)


