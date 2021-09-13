import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def flatten_list(list_of_lists: list):
    return [item for sublist in list_of_lists for item in sublist]


def children(root_):
    for child in root_:
        print(child.tag, child.attrib)
        children(child)


def get_name(str_):
    if str_ is None:
        return None
    if len(str_) == 0:
        return None
    try:
        return int(str_)
    except ValueError:
        lst = flatten_list([item.split('>') for item in str_.split('<')])
        for item in lst:
            try:
                return int(item)
            except ValueError:
                pass


def get_node_locations(root_, page='Page-1'):
    node_loc = {}
    vertexes = root_.findall(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@vertex='1']")
    for vertex in vertexes:
        name = get_name(vertex.attrib.get('value'))
        child = vertex.find("./mxGeometry")
        x = child.attrib.get('x')
        y = child.attrib.get('y')
        node_loc[name] = (x, y)
    return node_loc


def get_style_value_range(style: str, name: str):
    start = style.index('fillColor')
    value_start = start + style[start:].index('=') + 1
    end = start + style[start:].index(';')
    return value_start, end


def set_node_color(root_, node: int, color_hex: str, page='Page-1'):
    # for child in root_:
    #     if child.tag == "mxCell":
    #         if (name := get_name(child.attrib.get('value'))) is not None:
    #             if name == node:
    #                 print(f'Found {node}!')
    #             print(name)
    #     set_node_color(child, node, color)
    vertex = root_.find(f'./diagram[@name="{page}"]/mxGraphModel/root/mxCell[@vertex="1"]/[@value="{node}"]')
    style = vertex.attrib.get('style')
    start, end = get_style_value_range(style, 'fillColor')
    vertex.attrib['style'] = style[0:start] + color_hex + style[end:]


def set_node_color_from_voltage(root_, node: int, volt: float, vmin: float, vmax: float, page='Page-1'):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    hex_color = mcolors.to_hex(mapper.to_rgba(volt))
    # set_node_color(root_, node, hex_color, page=page)
    vertex = root_.find(f'./diagram[@name="{page}"]/mxGraphModel/root/mxCell[@vertex="1"]/[@value="{node}"]')
    style = vertex.attrib.get('style')
    start, end = get_style_value_range(style, 'fillColor')
    vertex.attrib['style'] = style[0:start] + hex_color + style[end:]
    # vertex.set('pf_res', 'test')


def color_nodes_from_voltdump(volt_dump_path, diagram_path, page='Page-1', phase='A', angle=False,
                              v_ll_base=4160, v_min=0.95, v_max=1.05):
    v_df = pd.read_csv(volt_dump_path, sep=',', header=1, index_col=0, parse_dates=True)
    node_names = np.array(v_df.index)
    prop = f'volt{phase.upper()}_mag'
    v_list = np.abs(np.array(v_df[prop], dtype=float))/(v_ll_base/np.sqrt(3))
    if angle:
        prop = f'volt{phase.upper()}_angle'
        v_list = np.abs(np.array(v_df[prop], dtype=float))
        v_min = np.min(v_list)
        v_max = np.max(v_list)
    tree = ET.parse(diagram_path)
    root = tree.getroot()
    set_node_color(root, 1, '#000000', page=page)
    for node_name, v in zip(node_names, v_list):
        node = int(node_name.split('_')[1])
        print(node)
        if angle:
            set_node_color_from_voltage(root, node, v, v_min, v_max, page=page)
        else:
            if v > 1e-3:
                # print(node)
                set_node_color_from_voltage(root, node, v, v_min, v_max, page=page)
            else:
                set_node_color(root, node, '#000000', page=page)
    tree.write(diagram_path)


def make_all_black(diagram_path, page='Page-1'):
    tree = ET.parse(diagram_path)
    root = tree.getroot()
    page1 = root.find(f"./diagram[@name='{page}']")
    vertexes = page1.findall("./mxGraphModel/root/mxCell[@vertex='1']")
    for vertex in vertexes:
        style = vertex.attrib.get('style')
        if style is not None:
            start, end = get_style_value_range(style, 'fillColor')
            vertex.attrib['style'] = style[0:start] + '#000000' + style[end:]
    tree.write(diagram_path)



if __name__ == '__main__':
    make_all_black('ieee123.drawio', page='voltA_mag')
    make_all_black('ieee123.drawio', page='voltB_mag')
    make_all_black('ieee123.drawio', page='voltC_mag')
    make_all_black('ieee123.drawio', page='voltA_angle')
    make_all_black('ieee123.drawio', page='voltB_angle')
    make_all_black('ieee123.drawio', page='voltC_angle')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='A', page='voltA_mag')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='B', page='voltB_mag')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='C', page='voltC_mag')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='A', angle=True, page='voltA_angle')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='B', angle=True, page='voltB_angle')
    color_nodes_from_voltdump('output_voltage.csv', 'ieee123.drawio', phase='C', angle=True, page='voltC_angle')
    # v_df = pd.read_csv('output_voltage.csv', sep=',', header=1, index_col=0, parse_dates=True)
    # node_names = np.array(v_df.index) 
    # v_a = np.abs(np.array(v_df['voltA_mag'], dtype=float)) / 2401.77
    # drawio_file_path = 'ieee123.drawio'
    # tree = ET.parse(drawio_file_path)
    # root = tree.getroot()
    # print(get_node_locations(root))
    # set_node_color_from_voltage(root, 1, 1.04, 0.95, 1.05)
    # for node_name, v in zip(node_names, v_a):
    #     node = int(node_name.split('_')[1])
    #     if v > 1e-3:
    #         print(node)
    #         set_node_color_from_voltage(root, node, v, 0.95, 1.05)
    #     else:
    #         set_node_color(root, node, '#000000')
    # tree.write(drawio_file_path)
