import streamlit as st
from vehicle_search.Model import VehicleSearch
from utils.Tool import StringUtils, ImageUtils
import cv2
import numpy as np
text_tool = StringUtils()
image_tool = ImageUtils()
model = VehicleSearch()
st.set_page_config(page_title="Vehicle Search", page_icon="🚗", layout="wide")
st.title("Vehicle Search")

st.markdown("This is a demo for vehicle search")
model.get_image()

with st.form("InputForm"):
    st.subheader("Parameters")
    conf_val = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    iou_val =  st.slider("IoU threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.subheader("Input text")
    input_text = st.text_input("Input text", "EK 090CN")
    st.subheader("Reload image?")
    st.caption("If you want to reload image, please check the box below")
    load_image = st.checkbox("Load image", value=False)
    st.subheader("Multiple search?")
    st.caption("If you want to search multiple image, please check the box below")
    multiple_search = st.checkbox("Multiple search", value=False)
    st.form_submit_button("Search")

st.subheader("Result")
if not multiple_search:
    with st.spinner("Searching..."):
        st.write("Input text: ", input_text)
        image, plate, text, xxyy, name = model.search(input_text, load_image=load_image, confidence_threshold=conf_val, iou_threshold=iou_val)

        if image is not None:
            st.success('Find success!', icon="✅")
            col1, col2 = st.columns([2,1])
            image = image_tool.post_processing_image(image, xxyy)

            floor, block = text_tool.post_process_text(name)
            col1.image(image, caption="Original image", use_column_width=True)
            col2.image(plate, caption="Plate image", use_column_width=True)
            col2.divider()
            col2.info(f"Plate number: {text}", icon="🚗")
            col2.divider()
            col2.info(f"Floor: {floor.capitalize()}", icon="🏠")
            col2.info(f"Block: {block.capitalize()}", icon="🏢")
        else:
            st.info('No vehicle found', icon="ℹ️")
else:
    with st.spinner("Searching..."):
        st.write("Input text: ", input_text)
        list_res = model.Multiple_search(input_text, load_image=load_image, confidence_threshold=conf_val, iou_threshold=iou_val)
        if len(list_res) > 0:
            st.success('Find success!', icon="✅")
            for image, plate, text, xxyy, name in list_res:
                col1, col2 = st.columns([2,1])
                image = image_tool.post_processing_image(image, xxyy)
                floor, block = text_tool.post_process_text(name)
                col1.image(image, caption="Original image", use_column_width=True)
                col2.image(plate, caption="Plate image", use_column_width=True)
                col2.divider()
                col2.info(f"Plate number: {text}", icon="🚗")
                col2.divider()
                col2.info(f"Floor: {floor.capitalize()}", icon="🏠")
                col2.info(f"Block: {block.capitalize()}", icon="🏢")
        else:
            st.info('No vehicle found', icon="ℹ️")



