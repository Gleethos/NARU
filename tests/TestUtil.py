from utility.lara_structures import create
import pprint

struc = create(15, 500, 39, 100, 10, return_info=True)

assert struc["dims"][2] == 214
assert struc["dims"][7] == 500
assert struc["dims"][4] == 328
assert struc["dims"][6] == 442
assert struc["dims"][8] == 430

assert struc["nodes"][3] == 17
assert struc["nodes"][7] == 39
assert struc["nodes"][8] == 33
assert struc["nodes"][14] == 1

# pprint.pprint(struc)

struc = create(depth=4, max_dim=200, max_paths=4, D_in=100, D_out=10, return_info=True)

# pprint.pprint(struc)

assert len(struc["layers"]) == 4
assert len(struc["layers"][0]) == 1
assert len(struc["layers"][3]) == 1
assert len(struc["layers"][2]) == 4
assert len(struc["layers"][1]) == 4
assert "sources" in struc["layers"][1][0]
assert "source" not in struc["layers"][1][0]
assert "sources" in struc["layers"][0][0]
assert len(struc["layers"][0][0]["sources"]) == 0
assert len(struc["layers"][3][0]["routes"]) == 0
assert len(struc["layers"][3][0]["sources"]) == 4
assert len(struc["layers"][1][0]["sources"]) == 1
assert len(struc["layers"][1][0]["routes"]) == 3

assert "route" in struc["layers"][1][0]["routes"][2]
assert "target_index" in struc["layers"][1][0]["routes"][2]

assert "source" in struc["layers"][1][1]["sources"][0]
assert "route" not in struc["layers"][1][1]["sources"][0]
assert "target_index" in struc["layers"][0][0]["routes"][0]

assert struc["layers"][1][1]["sources"][0]["target_index"] == 0
assert struc["layers"][2][0]["sources"][0]["target_index"] == 0

assert struc["layers"][2][1]["sources"][0]["target_index"] == 0
assert struc["layers"][2][2]["sources"][0]["target_index"] == 0
assert struc["layers"][2][2]["sources"][1]["target_index"] == 1
assert struc["layers"][2][2]["sources"][2]["target_index"] == 2
assert struc["layers"][2][3]["sources"][0]["target_index"] == 1
