import streamlit as st
import re


bot_msg_container_html_template = '''
<div style='background-color: #2C3333; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 20%; display: flex; justify-content: center ">
        <img src="https://w1.pngwing.com/pngs/278/853/png-transparent-line-art-nose-chatbot-internet-bot-artificial-intelligence-snout-head-smile-black-and-white-thumbnail.png" style="max-height: 50px; max-width: 50px; border-radius: 50%;">
    </div>
    <div style="width: 80%;">
        $MSG
    </div>
</div>
'''

user_msg_container_html_template = '''
<div style='background-color: #2C3333; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex'>
    <div style="width: 78%">
        $MSG
    </div>
    <div style="width: 20%; margin-left: auto; display: flex; justify-content: center;">
        <img src="https://storage.needpix.com/rsynced_images/user-2935527_1280.png" style="max-width: 50px; max-height: 50px; float: right; border-radius: 50%;">
    </div>
</div>
'''


def render_chat(**kwargs):
    """
    Handles is_user 
    """
    if kwargs["is_user"]:
        st.write(
            user_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)
    else:
        st.write(
            bot_msg_container_html_template.replace("$MSG", kwargs["message"]),
            unsafe_allow_html=True)

    if "figs" in kwargs:
        for f in kwargs["figs"]:
            st.plotly_chart(f, use_container_width=True)

