# ML_ops_stocks
The following repo consist of 6 diffrent files to make the whole pipeline more stable. The general objective 
of the project was to build an easy to use and simple version of stock prediction.

To collect data i've used the alpha vantage api, which works without any issues at all. For the "hosting" of data and models i've chosen to use hopsworks
which brought along some issues, but also solutions.

Atfer the data has been uploaded and saved into a csv file, preprocessing starts, which makes the data ready. Crucial step is the date alignment.

Herafter I upload the feature group which is all the data contained, where i've chosen to focus on date, open and closing price.

All in all the project takes advantage of predicting the future prices for the AMD stock only relying on future data and statistics within the stock itself.


