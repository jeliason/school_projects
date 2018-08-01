df<-read.csv("catalog sales data.csv")
# df <- newsales1
# str(df)
# dim(df)

##### targdol
# check distribution
barplot(table(df$targdol[df$targdol!=0]))

##### datelp6
# check distribution of datelp6
# add column of calendar dates of last purchase, check distribution
df$dm_lp <- format(as.Date(df$datelp6,"%m/%d/%Y"), "%m/%d")
df<-subset(df,select=-c(lpuryear))
barplot(table(df$dm_lp[df$targdol!=0]))
barplot(table(df$dm_lp))
sum(is.na(df)) # no other missing values!

# take year of last purchase from lp6 to make into new column
df$lpuryear2<-format(as.Date(df$datelp6,"%m/%d/%Y"), "%Y")

# print number of dates on either date
length(which(df$dm_lp=="03/01"))

length(which(df$dm_lp=="11/15"))

length(which(df$dm_lp!="11/15"&df$dm_lp!="03/01"&df$targdol!=0))

length(which(df$dm_lp!="11/15"&df$dm_lp!="03/01"&df$targdol==0))

length(which(df$targdol!=0))

# could treat datelp6 as factor (in the regression only, probably)? probably the best bet... meaning we'd exclude about 500 records
# (or we could treat those as factors too, though they'll probably end up being quite insignificant)

## DUMP into 6 month bins

x <- as.Date(df$datelp6,"%m/%d/%Y")

df$lp6_bin<-cut(x,breaks="6 months",labels=FALSE)

# find distribution of fall/spring purchases over last 4 years (for Ethan)
m<-max(df$lp6_bin)
length(which(df$lp6_bin==m-7))
df$datelp6[which(df$lp6_bin==m-5)]

##### datead6
# check distribution of datead6

df$dm_ad <- format(as.Date(df$datead6,"%m/%d/%Y"), "%m/%d")
# only targdol !=0
barplot(table(df$dm_ad[df$targdol!=0]))
# for all records
barplot(table(df$dm_ad))

# weird that there's this huge spike on this graph - the rest of these are too big to treat as factors
# not sure exactly what to do here (maybe dump into 6-month bins also)

# check that datead6 is before datelp6
idx<-which(as.Date(df$datead6,"%m/%d/%Y")>as.Date(df$datelp6,"%m/%d/%Y"))
bad_ad6<-data.frame("Add6"=df$datead6[idx],"Lp6"=df$datelp6[idx],"dist"=as.Date(df$datead6[idx],"%m/%d/%Y")-as.Date(df$datelp6[idx],"%m/%d/%Y"),"targdol"=df$targdol[idx])
hist(as.numeric(bad_ad6$dist))
dim(bad_ad6)
length(bad_ad6$dist[bad_ad6$dist>30])
length(bad_ad6$dist[bad_ad6$dist>30&bad_ad6$targdol!=0])

# so, could get rid of almost 400 records that have datead6 more than a month ahead of datelp6
# then for the rest, reassign datead6 to be datelp6 (or just get rid of all of them)

##### lpuryear2
# check distribution
barplot(table(df$lpuryear2))

# this looks pretty normal (no weird outliers, just generally ok-looking increase)
##### sales

# check distributions of sales
hist(df$slstyr)
hist(df$slslyr)
hist(df$slstyr,breaks=100)
hist(df$slslyr,breaks=100)
boxplot(df$slstyr)
# check that sales in a year doesn't coincide with zero orders, and vice versa

length(which(df$slstyr!=0&df$ordtyr==0))

length(which(df$slstyr==0&df$ordtyr!=0&df$targdol!=0))

length(which(df$slslyr!=0&df$ordlyr==0))

length(which(df$slslyr==0&df$ordlyr!=0&df$targdol!=0))

idx2<-which(df$slstyr==0&df$ordtyr!=0)
idx3<-which(df$slslyr==0&df$ordlyr!=0)
bad_slsty<-data.frame("slsty"=df$slstyr[idx2],"ordty"=df$ordtyr[idx2])
bad_slsly<-data.frame("slsly"=df$slslyr[idx3],"ordly"=df$ordlyr[idx3])

# as we can see, we'll probably have to eliminate some of these variables that are inconsistent (or reassign them, probably so that ordtyr=0)
# reassigning

df$ordtyr2<-df$ordtyr
df$ordtyr2[which(df$slstyr==0&df$ordtyr!=0)]=0

df$ordlyr2<-df$ordlyr
df$ordlyr2[which(df$slslyr==0&df$ordlyr!=0)]=0

##### recency variable

df$recency_numeric<-as.numeric(as.Date("2012-12-01")-as.Date(df$datelp6,"%m/%d/%Y"))
hist(df$recency)
hist(df$recency_numeric,breaks=100)

df$recency_factor<-max(df$lp6_bin)-df$lp6_bin



# in conclusion: new variables lpuryear2, lp6_bin, recency_numeric, recency_factor (use in regression). Don't use variables dm_ad, dm_lp, recency, lpuryear, datelp6.
# datead6 is unfixed for now (since I think it will be insignificant). If it ends up being significant, I'll change it.
# Also: use ordtyr2 and ordlyr2 instead of ordtyr and ordlyr

# Save work
write.csv(df, "Huazhen_Joel_Updated.csv")
