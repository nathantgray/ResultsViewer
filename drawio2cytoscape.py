import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def drawio2cytoscape(tree, xlen=30, ylen=30, grabbable=False, page='Page-1'):
    root = tree.getroot()
    cyto = []
    edges = []
    vertexes = root.findall(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@vertex='1']")
    for vertex in vertexes:
        id = vertex.attrib.get('id')
        name = vertex.attrib.get('value')
        child = vertex.find("./mxGeometry")
        x = child.attrib.get('x')
        y = child.attrib.get('y')
        cyto.append(
            {'data': {
                'id': id,
                'label': name
            },
             'position': {'x': int(float(x)), 'y': int(float(y))},
             # 'classes': 'terminal',
             'grabbable': grabbable
            })

    edge_elements = root.findall(f"./diagram[@name='{page}']/mxGraphModel/root/mxCell[@edge='1']")
    for edge in edge_elements:
        source = edge.attrib.get('source')
        target = edge.attrib.get('target')
        cyto.append({'data': {'source': source, 'target': target}})
    return cyto

def get_pages(tree):
    root = tree.getroot()
    page_list = []
    pages = root.findall("./diagram")
    for page in pages:
        page_list.append({'label': page.attrib.get('name'), 'value': page.attrib.get('name')})
    return page_list
