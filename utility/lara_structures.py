import math
from utility.lara_modules import Route, Source


def girth_from(length, start, max, end):
    g, mid = [], (length - 1) / 2
    for i in range(0, length):
        if math.fabs(mid - i) < 1:
            g.append(max)
        else:
            if i < mid:
                ratio = i / mid
                g.append((max * ratio) + start * (1 - ratio))
            else:
                ratio = (i - mid) / mid
                g.append((end * ratio) + max * (1 - ratio))
    return g


def create(
        depth=15,
        max_dim=500,
        max_paths=39,
        D_in=100,
        D_out=10,
        return_info=False,
        max_branch=3
):
    assert depth > 3
    dims = [math.floor(float(x)) for x in girth_from(depth, D_in, max_dim, D_out)]
    heights = [math.floor(float(x)) for x in girth_from(depth, 1, max_paths, 1)]
    conns = []
    for li in range(len(heights) - 1): conns.append(heights[li] * heights[li + 1])

    layers = []
    for li in range(depth):
        layer = []
        for ni in range(heights[li]):
            node = {"sources": [], "routes": [], "state": None, "active": False}
            if li < depth - 1:
                for bi in range(min(max_branch, heights[li + 1])):
                    node["routes"].append(
                        {
                            "route": Route(D_h=dims[li], D_in=dims[li + 1], D_out=dims[li]),
                            "target_index": int(math.fabs((bi + ni) % heights[li + 1]))
                        }
                    )
            layer.append(node)
        # Creating source modules:
        for ni in range(heights[li]):
            if li > 0:
                prev_layer = layers[li - 1]
                for prev_ni in range(len(prev_layer)):
                    prev_layer_node = prev_layer[prev_ni]
                    for route in prev_layer_node["routes"]:
                        if route["target_index"] == ni:
                            layer[route["target_index"]]["sources"].append(
                                {
                                    "source": Source(D_in=dims[li - 1], D_out=dims[li]),
                                    "target_index": prev_ni
                                }
                            )
        layers.append(layer)

    struct = {"layers": layers}
    if return_info:
        struct["dims"] = dims
        struct["nodes"] = heights
        struct["conns"] = conns

    return struct
