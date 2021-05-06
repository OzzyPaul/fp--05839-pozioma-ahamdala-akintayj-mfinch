import pandas as pd
import altair as alt
import numpy as np
from help_func import*
from sklearn.decomposition import PCA

##
# data preprocess
##

def data_preprocess(df):
    
    df = df.reset_index(drop=True)
    df["MR"] = df.apply(MR_df, axis=1)
    # df["corr"] = df.apply(lambda row: Correlation(row["MR"]), axis=1)
    df["Mag"] = df.apply(lambda row: Mag_dis(row["MR"]), axis=1)
    
    return df

##
# Join Prob
##

def gen_plot_df(r1):

    rtemp = r1["MR"].ravel()
    k, p = np.meshgrid(range(1,6), range(1,6))  
    plot_df = pd.DataFrame({"k":k.ravel(), "p":p.ravel(), "MR":rtemp, "cohort":r1["cohort"]})
    return plot_df


##
# Parent income prepocess for people
##

# Preprocess help funcs
def check_100(df):
    sum_check = 100 - df[["par_q1","par_q2","par_q3","par_q4","par_q5_mod","par_top10pc_mod","par_top1pc_mod","par_toppt1pc"]].sum(axis=1)
    
    df["par_q5_mod"] = df["par_q5_mod"] + sum_check
    
    return df

def func(row):
    q = list()
    
    q.append(np.repeat('par_q1', row['par_q1']).tolist())
    q.append(np.repeat('par_q2',row['par_q2']).tolist())
    q.append(np.repeat('par_q3',row['par_q3']).tolist())
    q.append(np.repeat('par_q4',row['par_q4']).tolist())
    q.append(np.repeat('par_q5',row['par_q5_mod']).tolist())
    q.append(np.repeat('par_top10pc',row['par_top10pc_mod']).tolist())
    q.append(np.repeat('par_top1pc',row['par_top1pc_mod']).tolist())
    q.append(np.repeat('par_toppt1pc',row['par_toppt1pc']).tolist())
    
    return q

#  https://stackoverflow.com/questions/716477/join-list-of-lists-in-python
def func1(col):
    result = list()
    for el in col:
        result.extend(list(el))
    return [result] 

def gen_people_df(row):
    
    rtemp = np.array(row["data"]).ravel()
    plot_df = pd.DataFrame({"data":rtemp, "cohort":row["cohort"]})
    return plot_df 

def Par_income_pre(df):

    parent_df = round(df[["par_q1","par_q2","par_q3","par_q4","par_q5","par_top10pc","par_top1pc","par_toppt1pc"]]*100)
    parent_df = pd.concat([df["cohort"],parent_df], axis=1)
    parent_df["par_q5_mod"] = parent_df["par_q5"] - parent_df["par_top10pc"]
    parent_df["par_top10pc_mod"] = parent_df["par_top10pc"] - parent_df["par_top1pc"]
    parent_df["par_top1pc_mod"] = parent_df["par_top1pc"] - parent_df["par_toppt1pc"]
    parent_df = check_100(parent_df)

    df_mod = parent_df.apply(lambda row: row.to_frame().apply(lambda col: func(col), axis =0), axis = 1)
    df_mod = pd.concat(list(df_mod), axis =1)
    df_mod = df_mod.apply(func1).T

    df_mod = pd.concat([df_mod,parent_df["cohort"]], axis =1)
    df_mod.columns = ["data","cohort"]
    df_mod = df_mod.apply(gen_people_df, axis=1)
    df_mod = pd.concat(list(df_mod)).reset_index()
    df_mod["index"] = df_mod["index"]+1

    return df_mod

def people_plot(df, select):
    # people 
    df_new1 = Par_income_pre(df)
    
    person_img = 'M1.7 -1.7h-0.8c0.3 -0.2 0.6 -0.5 0.6 -0.9c0 -0.6 -0.4 -1 -1 -1c-0.6 0 -1 0.4 -1 1c0 0.4 0.2 0.7 0.6 0.9h-0.8c-0.4 0 -0.7 0.3 -0.7 0.6v1.9c0 0.3 0.3 0.6 0.6 0.6h0.2c0 0 0 0.1 0 0.1v1.9c0 0.3 0.2 0.6 0.3 0.6h1.3c0.2 0 0.3 -0.3 0.3 -0.6v-1.8c0 0 0 -0.1 0 -0.1h0.2c0.3 0 0.6 -0.3 0.6 -0.6v-2c0.2 -0.3 -0.1 -0.6 -0.4 -0.6z'
 
    income_class8 = ["Lower class", "Working class", "Lower middle class", "Upper middle class", "Upper class", "Upper class (10%)","Upper class (1%)", "Upper class (0.1%)"]
    domains = ["par_q1","par_q2","par_q3","par_q4","par_q5", "par_top10pc", "par_top1pc", 'par_toppt1pc']
    for i in range(0,8): 
        df_new1.loc[df_new1["data"] == domains[i], "data"] = income_class8[i]
    
    shape_scale = alt.Scale(
        domain=income_class8,
        range = [person_img, person_img, person_img, person_img, person_img, person_img, person_img, person_img]
    )

    color_scale = alt.Scale(
        domain=income_class8,
        range=['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b', '#454B1B', '#00a235', '#45ff00']
    )

    base1 = alt.Chart(df_new1)

    people_chart = base1.transform_calculate(
        row ="ceil(datum.index/10)"
    ).transform_calculate(
        col="datum.index-datum.row*10"
    ).mark_point(filled=True, opacity = 1, size = 50).encode(
        alt.X('col:O', axis=None),
        alt.Y("row:O", axis=None),
        alt.Shape("data:N", legend = None, scale=shape_scale),
        alt.Color("data:N", legend = alt.Legend(title = "Income classes"), scale=color_scale)
    ).add_selection(select).transform_filter(select).properties(title = "Parent\'s Income")

    return df_new1, people_chart

##    
# https://stackoverflow.com/questions/55700724/controlling-stack-order-of-an-altair-area
# https://nextjournal.com/sdanisch/multi-view-composition
# https://stackoverflow.com/questions/31511997/pandas-dataframe-replace-all-values-in-a-column-based-on-condition/31512025
# http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2018/tutorials/t1b_plotting.html
# https://github.com/altair-viz/altair/issues/1826
##

def Joint_Prob_plot(df):
        
    year_min = df["cohort"].min()
    year_max = df["cohort"].max()
    slider = alt.binding_range(min=year_min, max=year_max, step = 1)
    select = alt.selection_single(name="year",fields=['cohort'],bind=slider, init ={'cohort':year_min} )
    
    df_new = df.apply(lambda row: gen_plot_df(row), axis = 1)
    df_new = pd.concat(list(df_new))
    
    income_class = ["Lower class", "Working class", "Lower middle class", "Upper middle class", "Upper class"]
    income_classy = ["Upper class","Upper middle class", "Lower middle class", "Working class", "Lower class"]
    
    for i in range(0,5): 
        df_new.loc[df_new["k"] == i+1, "k"] = income_class[i]
        df_new.loc[df_new["p"] == i+1, "p"] = income_class[i]


    base = alt.Chart(df_new)

    # chart on Joint Prob

    plot_scale = alt.Scale(type="pow", exponent=0.5, scheme = "greens", nice = True)
    color = alt.Color('MR:Q', scale = plot_scale, legend=alt.Legend(title="Mobility Rate"))

    joint_chart = base.mark_rect().encode(
        x=alt.X("p:N", 
            sort = income_class,
            axis=alt.Axis(title='Parent\'s Income Class')),
        y=alt.Y('k:N', 
            sort = income_classy,
            axis=alt.Axis(title='Children\'s Income Class')),
        color=color
        ).add_selection(select).transform_filter(select).properties(
            height=300, width=300, title='Children Vs Parents Income Distribution Heatmap')

    color_scale = alt.Scale(
        domain=income_class,
        range=['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b']
    )
    p_mag_1 = base.mark_bar().encode(
        y = alt.Y("p:N",
            sort = income_class, 
            axis=alt.Axis(title='Income Quintiles')),
        x = alt.X('sum(MR):Q', 
            scale=alt.Scale(domain=(0,1)), 
            axis=alt.Axis(title='Probabilities (p)')),
        color = alt.Color('k:N', 
            scale=color_scale,
            legend = alt.Legend(title=None)),
        order=alt.Order(aggregate='sum', type="quantitative", sort='descending')
        ).add_selection(select).transform_filter(select).properties(
            height=300, width=300, title='Parent\'s Income Probability Distribution')

    k_mag_1 = base.mark_bar().encode(
        y = alt.Y("k:N", 
            sort = income_classy,
            axis=alt.Axis(title='Income Quintiles')),
        x = alt.X('sum(MR):Q', 
            axis=alt.Axis(title='Probabilities (k)'),
            scale=alt.Scale(domain=(0,1))),
        color = alt.Color('p:N', 
            scale=color_scale,
            legend = alt.Legend(title=None), 
            sort=alt.EncodingSortField('p', order='ascending')),
            order=alt.Order(aggregate='sum', type="quantitative", sort='descending')
        ).add_selection(select).transform_filter(select).properties(
            height=300, width=300, title='Children\'s Income Probability Distribution')
        
    _, people_c = people_plot(df,select)
    people_c = people_c.properties(width=300, height = 300)

    JPP_explanation = """The probability distribution charts below show actual and forecasted household as well as children's income classes.
                        The horizontal axes for both charts indicate the probabilities (forecast) of the parents/children belonging to each of the five income classes on the vertical axes.
                        The stacks on each vertical bar show the actual income distributions within the forecasted classes i.e how many households/children actually fall in the forecasted class. The income classes are color-coded.
                    """
    st.write(JPP_explanation)
    
    return (p_mag_1 & k_mag_1 & joint_chart & people_c).resolve_scale(color='independent').configure_axis(titleFontWeight = "normal")
    
##
# cluster helper funcs
##

def data_pre(source):
    
    s1 = source[source["k"]>source["p"]]
    s1 = s1.groupby(['p']).sum()
    s1['name'] = source['name']
    s1['tier_name'] = source['tier_name']
    s1['k'] = source['k']

    s2 = source[source["k"]==source["p"]]
    s2 = s2.groupby(['p']).sum()
    s2['name'] = source['name']
    s2['tier_name'] = source['tier_name']
    s2['k'] = source['k']

    s3 = source[source["k"]<source["p"]]
    s3 = -1*s3.groupby(['p']).sum()
    s3['name'] = source['name']
    s3['tier_name'] = source['tier_name']
    s3['k'] = source['k']

    s1["type"] = "pos"
    s1["type_num"] = 1
    s2["type"] = "same"
    s2["type_num"] = 0
    s3["type"] = "neg"
    s3["type_num"] = -1

    s = pd.concat([s1,s2,s3])
    s = s.reset_index()
    
    s['MR_abs']= round(abs(s['MR'])*100,2)
    
    return s

##
##
def gen_plot_df_name(r1):

    rtemp = r1["MR"].ravel()
    k, p = np.meshgrid(range(1,6), range(1,6))  
    plot_df = pd.DataFrame({"k":k.ravel(), "p":p.ravel(), "MR":rtemp, "name":r1["name"], "tier_name":r1["tier_name"]})
    
    return data_pre(plot_df)

def distance(df1, df2):
    dneg = df1["neg"] - df2["neg"]
    dpos = df1["pos"] - df2["pos"]
    dsame = df1["same"] - df2["same"]
    return np.sqrt(dneg*dneg+dpos*dpos+dsame*dsame)

def filter(df, t_name,U_name):
    df_new = df.loc[df['tier_name']==t_name]
    df_new = df_new.loc[df_new["name"]!=U_name]
    df_new = df_new.sort_values(by=["distance","name"], ascending=True)[:10]
    return df_new

def filter_name(df, name):
    return df.loc[df["name"]==name]

##
# https://stackoverflow.com/questions/59381202/python-altair-condition-color-opacity
##

def cluster_plot(data, U_df, U_Name):

    cp_note = """ 
                The two plots below - the scatter plot and the horizontal stacked bar chart show how the children that attend colleges/universities turn out in the income 
                mobility ladder based on the tier of higher learning that they attend(ed).

                The scatter plot clusters the schools based on the college tier that they belong and consequently displays the student income mobility direction in the haorizontal stacked bar chart below it.

                What is interesting here is this bar chart also indicates the proportion of this income class movement, down to each income class, while also color coding the movements as follows: 
                * Green: The student/child moved upwards from their household's income class
                * Yellow: The student/child maintains the same income level as their household
                * Red: The student/child drops from their houshold's income class
            """

    st.write(cp_note)

    U_tier = data.groupby(["name","tier_name"]).mean()
    U_tier = U_tier.reset_index()
    U_tier = data_preprocess(U_tier)
    
    source = U_tier.apply(gen_plot_df_name, axis =1)
    source = pd.concat(list(source), axis = 0)
    source = source.reset_index()
    source_new = source[["name","MR","type","tier_name"]].groupby(["name","type","tier_name"]).sum()

    s = source_new.stack().to_frame()
    s = s.unstack(1)
    s.columns = s.columns.droplevel()
    source_new = s.reset_index()

    features = ["neg","pos","same"]
    x = source_new.loc[:,features].values
    y = source_new.loc[:,["name"]].values
    pca = PCA(n_components=2)
    prinComp = pca.fit_transform(x)
    prinDf = pd.DataFrame(data=prinComp, columns=['F1', 'F2'])
    df_name = source_new[['name','tier_name']]
    x_df = pd.DataFrame(data=np.round(100*(x),2), columns=features)
    finaldf = pd.concat([prinDf, df_name,x_df], axis=1)

    int_val = U_df['tier_name'].unique()[0]
    sel = alt.selection_single(fields=["tier_name"], bind='legend', init={'tier_name':int_val}, nearest=True, empty="none")
    color = alt.condition(sel, alt.Color('tier_name:N', scale=alt.Scale(scheme='set1')), alt.value('lightgray') )
    opacity = alt.condition(sel, alt.value(1.0), alt.value(0.25))

    d = university_df(finaldf, U_Name)
    finaldf["distance"] = finaldf.apply(lambda x: distance(x,d), axis = 1)
    t_name = finaldf["tier_name"].unique()

    df_new = pd.DataFrame()
    for name in t_name:
        df_new = pd.concat([df_new,filter(finaldf,name, U_Name)])
    
    df_new = df_new.reset_index()
    name1 = list(df_new["name"].unique())
        
    s_new = pd.DataFrame()
    d_new = pd.DataFrame()
    for n in name1:
        s_new = pd.concat([s_new,filter_name(source,n)], axis = 0)
        d_new = pd.concat([d_new,filter_name(finaldf,n)], axis = 0)
    
##
## Cluster Plot
## 

    cluster_plot = alt.Chart(finaldf).mark_point(size=50, filled=True, fillOpacity=1).encode(
        alt.X("F1:Q", sort = alt.Sort('descending'), axis=None),
        alt.Y("F2:Q", axis = None),
        color = color,
        opacity=opacity,
        tooltip=['name','pos','same','neg', "tier_name"]
    ).add_selection(sel)

##
## Cross Plot
##

    cross_u = alt.Chart(d).mark_point(size = 250, shape = 'diamond', filled = True).encode( 
    alt.X("F1:Q", sort = alt.Sort('descending'), axis = None),
    alt.Y("F2:Q", axis = None),
    color = alt.Color('tier_name:N', scale=alt.Scale(scheme='set1'))
    )

    cross_y = alt.Chart(d_new).mark_point(size = 250, shape = 'cross', filled = True).encode( 
    alt.X("F1:Q", sort = alt.Sort('descending'), axis =None ),
    alt.Y("F2:Q", axis = None),
    color = alt.Color('tier_name:N', 
        scale=alt.Scale(scheme='set1'),
        legend=alt.Legend(title=None)
        )
    ).transform_filter(sel)

 
##
## Bar Plot
##
    
    U_data = university_df(source, U_Name)
    new_U_data = pd.DataFrame()

    for n in t_name:
        temp = U_data.replace({"tier_name":list(U_data["tier_name"])}, {"tier_name":n}) 
        new_U_data = pd.concat([new_U_data,temp])

    s_new = pd.concat([new_U_data,s_new])
    
    color_scale1 = alt.Scale(
        domain=["pos","same","neg"],
        range=["green","orange","red"]
    )

    # https://stackoverflow.com/questions/52223358/rename-tooltip-in-altair
    s_new = s_new.reset_index()
    income_class = ["Lower class", "Working class", "Lower middle class", "Upper middle class", "Upper class"]
    
    for i in range(0,5): 
        s_new.loc[s_new["p"] == i+1, "p"] = income_class[i]

    bar_chart = alt.Chart(s_new).mark_bar().encode(
        y = alt.Y("name:N", 
            axis=alt.Axis(title=None)
            ),
        x = alt.X("sum(MR):Q",
            axis=alt.Axis(title="Mobility Rate")
            ),
        color = alt.Color('type:N', 
            scale = color_scale1,
            legend=alt.Legend(title=None)
            ),
        tooltip=[alt.Tooltip("sum(MR_abs)",title="Mobility Rate"), alt.Tooltip('p', title="Parent's Income Quintiles")]
    ).transform_filter(sel)
    
    
    cluster = cluster_plot+cross_u+cross_y
    
    return (cluster & bar_chart).resolve_scale(color='independent').resolve_legend(color='independent')