import json

in1 = 'C:/Users/fareed/PycharmProjects/tf_project/inc/txt_part_20_1799mapping.txt' #the file that maps nodes numbers and names(it is one of preProcess results just save it as txt
                    #and remove JSON beginning and end paranthesis)
in2 = 'C:/Users/fareed/PycharmProjects/tf_project/inc/with_edges_metis.graph.part.2' # metis generated output
out = 'C:/Users/fareed/PycharmProjects/tf_project/inc/metis_2.json' # the file that will contain the final mapping which you use to do the placement

pairs = {}
with open(in1, "r") as f, open(in2, "r") as f2:
    for line, line2 in zip(f, f2):
        pairs[line.split(":")[0].replace('\"', '').replace('\n', '')] = line2.replace('\n', '')


outfile = "C:/Users/fareed/PycharmProjects/tf_project/inc/metis_2.place"

with open(outfile, "w+") as f2:
	for key, val in pairs.items():
		f2.write(str(key) + " " + str(val) + "\n")



