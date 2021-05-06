##
# import files
##
import pandas as pd
import streamlit as st
import numpy as np
from help_func import*
from plot_func import*

#####
# main streamlit app
#####

st.markdown("test")

opening_paragraph = """
                      Are children from households with higher income more likely to earn more in their adulthood? 
                      Typically, household income determines the types of opportunities that are available to a child.
                      Does household income also affect the types of colleges that kids get into?""" 

first_paragraph = """In the US, children from high income families tend to attend more prestigious colleges. The mean income 
                      of households varies significantly between high ranking and low ranking colleges. The mean income of parents of
                      kids in Ivy League schools is significantly higher than those of non-selective and even two year colleges. College data from 1999 to 2003 
                      shows that mean parent income in Ivy Leagues is around $430000 while that of Two year for-profit colleges is around $63000"""

# Can we, perhaps have a viz that shows household income distribution for various college tiers. 
df = pd.read_csv("mrc_table2.csv")
df_reduced = df[['name', 'type', 'tier', 'tier_name', 'iclevel', 'par_mean', 'par_median', 'par_rank', 'k_mean', 'k_median', 'k_rank', 'k_median_nozero', 'k_0inc', 'k_q1', 'k_q2', 'k_q3', 'k_q4', 'k_q5']]
df_reduced_par = df_reduced[['tier_name', 'par_mean', 'par_median', 'par_rank', 'k_q1', 'k_q2', 'k_q3', 'k_q4', 'k_q5']]

df_reduced_par_cum = df_reduced_par.groupby(df['tier_name']).mean()
df_reduced_par_cum = df_reduced_par_cum.drop('Attending college with insufficient data')
df_reduced_par_cum = df_reduced_par_cum.drop('Not in college between the ages of 19-22')
df_reduced_par_cum = df_reduced_par_cum.reset_index()
df_reduced_par_cum['College Tier'] = df_reduced_par_cum['tier_name']
df_reduced_par_cum['Mean Parent Income in $'] = (df_reduced_par_cum['par_mean']).astype(int)

df_reduced_par_cum['First Quintile'] = df_reduced_par_cum['k_q1'] * 100
df_reduced_par_cum['Second Quintile'] = df_reduced_par_cum['k_q2'] * 100
df_reduced_par_cum['Third Quintile'] = df_reduced_par_cum['k_q3'] * 100
df_reduced_par_cum['Fourth Quintile'] = df_reduced_par_cum['k_q4'] * 100
df_reduced_par_cum['Fifth Quintile'] = df_reduced_par_cum['k_q5'] * 100

brush = alt.selection_single()

par_college_tier = alt.Chart(df_reduced_par_cum).mark_bar().encode(
    alt.X('tier_name:N', title=' ' , sort='-y', axis=alt.Axis(labels=False)),
    alt.Y('par_mean:Q', title='Mean Yearly Income of Parents in $'), tooltip=['College Tier', 'Mean Parent Income in $'],  color=alt.Color(
        "College Tier:N",
        legend=None)
    ).interactive().properties(height=400, width=600, title="Mean Yearly Income of Parents for Each College Tier").add_selection(
    brush
)



par_income_dist = alt.Chart(df_reduced_par_cum).transform_fold(
  ['First Quintile', 'Second Quintile', 'Third Quintile', 'Fourth Quintile', 'Fifth Quintile'],
  as_=['Quintile', 'Percentage of Parents']
).mark_bar().encode(
    x=alt.X('Percentage of Parents:Q'),
    y=alt.Y('Quintile:N', sort=['Fifth Quintile', 'Fourth Quintile', 'Third Quintile', 'Second Quintile', 'First Quintile' ]), color='Quintile:N'
).transform_filter(
    brush).properties(height=100, width=600, title="Percentage of Parents in Each Income Quintile")



first_paragraph_cont = """The percentage of students from the lowest income quintile that attend Ivy League Colleges are also low"""









second_paragraph = """Sadly, the type of colleges attended impacts what would be earned in adulthood. Graduates of top-tier colleges are likely
                      to earn more than graduates of lower-tier colleges. The median income of graduates from colleges reduces with the college tier reduces """

#Can we have a viz that shows median income of college graduates based on college tier
second_paragraph_cont = """How likely, then, is a child from a low income household to move up the income ladder?"""

#Perhaps a viz that starts from a node and spreads out to the various income percentile

third_paragraph = """Does college length also matter? How do graduate earnings from 2 and 3-year colleges compare to earnings from 4-year colleges?"""

#Can we have a chart that shows income distribution of graduates from these college types here?

third_paragraph_cont = """But 2 and 3 year colleges tend to have more kids from lower income households"""
#Can we have some sort of viz for this? Say something than branches out from low income households to college duration


fourth_paragraph = """But what can we do to ensure that kids from households with higher incomes can earn as much as those from lower income households"""


#Ki la le se??? -- What can we do?

fifth_paragraph = """To change the narrative, colleges can consider:
                      """

fifth_paragraph_cont = """1. Extending student recruitment and info sessions to students from low income and disadvantaged communities."""
fifth_paragraph_cont_1 = """2. Reducing legacy admissions and benefits for incoming students"""

st.title("Intergenerational Mobility and US Colleges")
st.write("\n\n\n\n\n\n")
st.write(opening_paragraph)
st.write(first_paragraph)
st.write(par_college_tier & par_income_dist)

st.write(second_paragraph)

st.write(second_paragraph_cont)

st.write(third_paragraph)

st.write(third_paragraph_cont)

st.write(fourth_paragraph)

st.write(fifth_paragraph)
st.write(fifth_paragraph_cont)
st.write(fifth_paragraph_cont_1)

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
