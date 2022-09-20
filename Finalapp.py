import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict



st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("1K Classifer(Dictionary)")
st.sidebar.write("Using Efficientnet_V2_L to determine what the image is......")
import streamlit as st




st.sidebar.write("")

img_source = st.sidebar.radio("Select image source",
                              ("Upload a picture", "Take a photo"))
if img_source == "Upload a picture":
    img_file = st.sidebar.file_uploader("Select image", type=["png", "jpg", "jpeg"])
elif img_source == "Take a photo":
    img_file = st.camera_input("Take a photo")

if img_file is not None:
    with st.spinner("Estimating..."):
        img = Image.open(img_file)
        st.image(img, caption="subject image", width=400)
        st.write("")
        

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("The results will be as follows ... ")
        n_top = 3  
        for result in results[:n_top]:
            st.write(" This picture with a probability of " + str(round(result[1]*100, 1))  + "%" + " is " + result[0] )

        # 円グラフの表示
        pie_labels = [result[0] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[1] for result in results[:n_top]]
        pie_probs.append(sum([result[1] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.5, "edgecolor":"white"}
        textprops = {"fontsize":7,'color': "white", 'weight': "bold"}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%1.1f", wedgeprops=wedgeprops)  
   
        fig.patch.set_facecolor('0E1117') 
        st.pyplot(fig)
        
        # 1番目のkeywordをwikipediaで調べる
        st.write('Let’s find out the details by Wikipedia！')
        n_top = 1
        for result in results[:n_top]:
            url = "https://en.wikipedia.org/wiki/"+ result[0]
            link = st.write(url)
  
        
#音楽挿入
import streamlit as st
import streamlit.components.v1 as stc
import base64
import time


button = st.sidebar.button('Play Music')

if button:

    audio_path1 = 'dreams.mp3' 

    audio_placeholder = st.empty()

    file_ = open(audio_path1, "rb")
    contents = file_.read()
    file_.close()

    audio_str = "data:audio/ogg;base64,%s"%(base64.b64encode(contents).decode())
    audio_html = """
                    <audio autoplay=True>
                    <source src="%s" type="audio/ogg" autoplay=True>
                    Your browser does not support the audio element.
                    </audio>
                """ %audio_str

    audio_placeholder.empty()
    time.sleep(0.5)
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

button = st.sidebar.button('Stop Music')
    
    



