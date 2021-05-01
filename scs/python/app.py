##
# import files
##

import streamlit as st
import numpy as np
from help_func import*
from plot_func import*

#####
# main streamlit app
#####

st.markdown("test")


name = "mrc_table3" 
data = read_file(name).dropna() # remove nan data rows

# https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value

SCHOOLS = data['name'].unique()
index_CMU = np.where(SCHOOLS == "Carnegie Mellon University")[0][0]

# https://discuss.streamlit.io/t/select-an-item-from-multiselect-on-the-sidebar/1276/2
SCHOOL_SEL = st.selectbox('Select University', SCHOOLS, index = int(index_CMU))

st.markdown(SCHOOL_SEL)

pre_data = data_preprocess(university_df(data, SCHOOL_SEL))

# plot Joint
st.markdown("text regarding plot below")
JPP=Joint_Prob_plot(pre_data)
st.write(JPP)

# cluster plot
st.markdown("more text for plot below")
cp=cluster_plot(data,pre_data,SCHOOL_SEL)
st.write(cp)

# https://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/
