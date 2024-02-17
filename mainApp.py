import streamlit as st  # streamlit==1.30.0
import easyocr
import cv2
                    # opencv-contrib-python        4.5.5.62
                    # opencv-python                4.5.4.60
                    # opencv-python-headless       4.5.1.48
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------------------------
# mainApp.py (02/16/2024) - Ver0_1 9JY_st_easyOCR0_1_stio.py)
# Changed the name from JY_st_easyOCR0_1.py - Ver0_1 (02/16/2024)
# - IMPORTANT UPDATE:
# (1) When attempted to run the app on streamlit server, the  error appeared:
# "img_np=cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)"
# while using opencv-python-headless==4.5.5.64 or 4.9.0.80.
# Couldn't find a solution as of 02/16/2024.
# Decided to rewrite the code without using cv2.cvtColor(). --> Fixed the bug when running the code
# on the streamlit server.
# (2) Used a path in the form of r"sample_images/image_demo_licnesePlate.png";
# -----------------------------------------------------------------------------------------------

CODE_TITLE='🔡 OCR'
CODE_VER='Ver 0_1 -- Using easyOCR'


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> START OF MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
st.title(CODE_TITLE)
st.caption(CODE_VER)

# -----------------------------------------------------------------------------------------------------------------
# init_easyOCR() - Ver0 (02/15/2024)
# - Inputs:
# (1) lang_list:<list of str> languages available in easyOCR; e.g., ['es'] for English;
# - Return:
# (1) reader:<the easyOCR reader instance>
# -----------------------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner='Loading easyOCR engine...')
def init_easyOCR(lang_list=['es']):
    reader=easyocr.Reader(lang_list)
    return reader

# # UNIVERSAL VARIABLES;
selectbox_options=[
                    # 'Scroll down to select an image',
                    'License Plate','Receipt','Prescription']
sample_imgs_dict={
                'Scroll down to select an image': None,
                'License Plate': r"sample_images/image_demo_licnesePlate.png",
                'Receipt': r"/sample_images/img_demo_receipt.jpg" ,
                'Prescription': r"/sample_images/medical-prescription-ocr.webp",
                  }
radio_btn_options=['🖼️ Upload an Image','🗃️ Select an Image Sample']

# # Load OCR engine;
reader=init_easyOCR(['es'])

# # Initialization
img_uploaded_name=None 
img_uploaded=None 

container1=st.sidebar.container()
container2=st.sidebar.container()
container3=st.sidebar.container()

# # Choose options betwen sample images or image upload;
radio_btn=container1.radio('⭐ Choose a method:',
                    radio_btn_options,
                   index=0,
                   key='radiobtn_chooseamethod0',
                   )

img_width=480
with container3:
    st.markdown('')
    with st.expander('Image Width',expanded=False):
        img_width=st.slider('Choose the image width:', min_value=320,max_value=800, value=480, step=20,
                            help='Default = 480',
                            key='slider_imgwidth0',
                            )

if radio_btn==radio_btn_options[1]:  #'Select an Image Sample'
    st.subheader('🗃️ Sample Image')
    sample_image=container2.selectbox('🗃️ Choose an image:',
                              selectbox_options,
                        #   ['License Plate','Receipt','Prescription'],
                              index=None,
                              placeholder='Scroll down to select an image',
                              key='selectbox_imguploaded0',
                              )
    
    if sample_image is not None:  # That is, a user selects an image from st.sidebar.selectbox() other than its placeholder value.
        img_uploaded_name=sample_imgs_dict[sample_image]    

    # if img_uploaded_name != '':      
                                # img_np=np.array(img)
        img_np=cv2.imread(img_uploaded_name)
        # img_np=img_np[:,:,::-1]
        # img_np=cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        st.image(img_np, width=img_width,caption=f'{sample_image}')

# # Upload an input image
if radio_btn==radio_btn_options[0]:  # '🖼️ Upload an Image'
    img_uploaded=container1.file_uploader('🖼️ Upload an image file', type=['jpg','jpeg','png','webp'])


if (radio_btn==radio_btn_options[0]) and (img_uploaded is not None):  # '🖼️ Upload an Image','Select an Image Sample'
    st.subheader('🖼️ Uploaded Image:')
    st.image(img_uploaded, width=img_width, caption=f'{img_uploaded.name}')
    img=Image.open(img_uploaded)
    # # Convert a file from st.file_uploader() to cv2 format;
                            # img_np=np.array(img)
    # img_np=cv2.imread(img_uploaded.name)
    # img_np=img_np[:,:,::-1]
    # img_np=cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    img_uploaded_name=img_uploaded.name

with st.form('my form'):
    submitted=st.form_submit_button('🚀 Run OCR')
    if submitted:
        if (img_uploaded is None) and (img_uploaded_name is None):
            st.warning('Please upload an image file in the left side panel.')
        else:
            # st.image(img_uploaded, caption=f'{img_uploaded.name}')
            # img=Image.open(img_uploaded)

            # # # Convert a file from st.file_uploader() to cv2 format;
            #                         # img_np=np.array(img)
            # img_np=cv2.imread(img_uploaded.name)
            # img_np=cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
            if submitted:
                show_info1=st.empty()
                show_info1.info('Applying OCR to the image (it may take a while)...')
                result=reader.readtext(img_uploaded_name)
                show_info1.empty()

                with st.expander('Results: text and probability', expanded=False):
                    for (bbox,text,prob) in result:
                        st.markdown(f'Text: {text}, Probability: {prob}')
                    for res in result:
                        st.markdown(f'Text: {res[1]}, Coordinates: {res[0]}')

                # font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                font=cv2.FONT_HERSHEY_SIMPLEX
                for reslt in result:
                    # # should take integers of a tuple - Ver0_1 (02/15/2024)
                    top_left=tuple(reslt[0][0])  # e.g., (102, 606)
                    top_left=tuple(map(int, top_left))
                    bottom_right=tuple(reslt[0][2])
                    bottom_right=tuple(map(int, bottom_right))
                    text=re.sub("""[."'}]""","",reslt[1])
                    img_np=cv2.rectangle(img_np,top_left,bottom_right, 
                                        (0,255,0), # (0,255,100),
                                        2)
                    # img_np=cv2.putText(img_np,text,bottom_right,font, 1.5, (0,255,0),
                    #                 2,  # 4
                    #                 cv2.LINE_AA
                    #                 )

                                    # # Rotate image colors from BGR to RGB -- NOT QUITE WORKING (Ver0) (02/15/2024)
                                    # img_rgb=img_np[:,:,::-1]
            
                st.subheader('📋 Result:')
                st.image(img_np, channels='RGB')
                c1,c2,c3=st.columns((6,1,2))
                c1.markdown('📜*Recognized Text:*')
                c3.markdown('🎯 *Confidence Level:*')
                for (_,text,prob) in result:
                    c1.markdown(text)
                    c3.markdown(f'{prob*100:0.2f}%')
                st.divider()
                with st.expander('View the image in a graph',expanded=False):
                    plt.figure(figsize=(10,10))
                    plt.imshow(img_np)
                    st.pyplot(plt.gcf())
                    st.markdown('')
    # ----------------------------- END OF st.form() -----------------------------------

for i in range(12):
    st.sidebar.markdown('')

# # Coloring st.button() 
with st.sidebar.expander('🎨 Color for buttons', expanded=False):
# with container_colorBtn:
    primaryColor=st.color_picker('Click color to change', "#008CF9")  # "#00f900")
s = f"""
<style>
div.stButton > button:first-child {{ border: 3px solid {primaryColor}; border-radius:10px 10px 10px 10px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)


with st.sidebar.expander('💬 Note:',expanded=True):  #  📝 
    msg="""
        Coming soon: Look out for updates on additional OCR algorithms like Tesseract and Keras.
"""
    st.markdown(msg)

for i in range(14):
    st.sidebar.markdown('')
    st.markdown('')
# ------------------------------------ END OF CODE -----------------------------------------------
        
