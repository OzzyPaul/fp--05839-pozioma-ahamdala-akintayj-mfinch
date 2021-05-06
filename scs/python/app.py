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
data_raw = read_file(name)
data = data_raw[["super_opeid","cohort","name", \
    "type","tier","tier_name",\
    "par_q1","par_q2","par_q3","par_q4","par_q5", \
    "par_top10pc","par_top5pc","par_top1pc","par_toppt1pc", \
    "kq1_cond_parq1", "kq2_cond_parq1", "kq3_cond_parq1", "kq4_cond_parq1", "kq5_cond_parq1", \
    "kq1_cond_parq2", "kq2_cond_parq2", "kq3_cond_parq2", "kq4_cond_parq2", "kq5_cond_parq2", \
    "kq1_cond_parq3", "kq2_cond_parq3", "kq3_cond_parq3", "kq4_cond_parq3", "kq5_cond_parq3", \
    "kq1_cond_parq4", "kq2_cond_parq4", "kq3_cond_parq4", "kq4_cond_parq4", "kq5_cond_parq4", \
    "kq1_cond_parq5", "kq2_cond_parq5", "kq3_cond_parq5", "kq4_cond_parq5", "kq5_cond_parq5" \
]].dropna() # remove nan data rows

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
st.altair_chart(JPP, use_container_width=True)

# cluster plot
st.markdown("more text for plot below")
cp=cluster_plot(data,pre_data,SCHOOL_SEL)
st.altair_chart(cp, use_container_width=True)

# https://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/
