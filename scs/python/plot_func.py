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
        
    domains = ["par_q1","par_q2","par_q3","par_q4","par_q5", "par_top10pc", "par_top1pc", 'par_toppt1pc']

    shape_scale = alt.Scale(
        domain=domains,
        range = [person_img, person_img, person_img, person_img, person_img, person_img, person_img, person_img]
    )

    color_scale = alt.Scale(
        domain=domains,
        range=['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b', '#54a200', '#00a235', '#45ff00']
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
        alt.Color("data:N", legend = None, scale=color_scale)
    ).add_selection(select).transform_filter(select)

    return people_chart

##    
# https://stackoverflow.com/questions/55700724/controlling-stack-order-of-an-altair-area
# https://nextjournal.com/sdanisch/multi-view-composition
##

def Joint_Prob_plot(df):
        
    year_min = df["cohort"].min()
    year_max = df["cohort"].max()
    slider = alt.binding_range(min=year_min, max=year_max, step = 1)
    select = alt.selection_single(name="year",fields=['cohort'],bind=slider, init ={'cohort':year_min} )
    
    df_new = df.apply(lambda row: gen_plot_df(row), axis = 1)
    df_new = pd.concat(list(df_new))

    base = alt.Chart(df_new)

    # chart on Joint Prob

    plot_scale = alt.Scale(type="pow", exponent=0.5, scheme = "greens", nice = True)
    color = alt.Color('MR:Q', scale = plot_scale)

    joint_chart = base.mark_rect().encode(
        x="p:O",
        y=alt.Y('k:O',
            sort=alt.EncodingSortField('k', order='descending')),
            color=color
        ).add_selection(select).transform_filter(select).properties(height=200, width = 200)

    color_scale = alt.Scale(
        domain=['1','2','3','4','5'],
        range=['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b']
    )
    p_mag_1 = base.mark_bar().encode(
        x = alt.X("p:O"),
        y = alt.Y('sum(MR):Q', scale=alt.Scale(domain=(0,1))),
        color = alt.Color('k:O', 
            scale=color_scale,
            legend = None),
        order=alt.Order(aggregate='sum', type="quantitative", sort='descending')
        ).add_selection(select).transform_filter(select).properties(height = 150, width=200)


    k_mag_1 = base.mark_bar().encode(
        y = alt.Y("k:O", 
            sort=alt.EncodingSortField('k', order='descending') ),
        x = alt.X('sum(MR):Q', scale=alt.Scale(domain=(0,1))),
        color = alt.Color('p:O', 
            scale=color_scale, 
            sort=alt.EncodingSortField('p', order='ascending'),
            legend = None),
            order=alt.Order(aggregate='sum', type="quantitative", sort='descending')
        ).add_selection(select).transform_filter(select).properties(height=200, width = 150)


    people_c = people_plot(df,select).properties(width=300, height = 300)

    return (p_mag_1 & (joint_chart | k_mag_1) & people_c).resolve_scale(color='independent')
    
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


##
# https://stackoverflow.com/questions/59381202/python-altair-condition-color-opacity
#
##

def cluster_plot(data, U_df, U_Name):

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
    
##
## Cluster Plot
## 

    cluster_plot = alt.Chart(finaldf).mark_point(size=50, filled=True, fillOpacity=1).encode(
        alt.X("F1:Q", sort = alt.Sort('ascending')),
        alt.Y("F2:Q"),
        color = color,
        opacity=opacity,
        tooltip=['name','pos','same','neg', "tier_name"]
    ).add_selection(sel)

##
## Cross Plot
##

    cross_u = alt.Chart(university_df(finaldf, U_Name)).mark_point(size = 250, shape = 'diamond', filled = True).encode( 
    alt.X("F1:Q", sort = alt.Sort('ascending')),
    alt.Y("F2:Q"),
    color = alt.Color('tier_name:N', scale=alt.Scale(scheme='set1'))
    )

##
## Bar Plot
##
    
    color_scale1 = alt.Scale(
        domain=["pos","same","neg"],
        range=["green","orange","red"]
    )

    bar_chart = alt.Chart(source).mark_bar().encode(
        column = "tier_name:N",
        y = alt.Y("name:N"),
        x = "sum(MR):Q",
        color = alt.Color('type:N', scale = color_scale1),
        tooltip=["sum(MR_abs)", 'p']
    ).transform_filter(sel)


    return ((cluster_plot+cross_u) & bar_chart).resolve_scale(color='independent').resolve_legend(color='independent')