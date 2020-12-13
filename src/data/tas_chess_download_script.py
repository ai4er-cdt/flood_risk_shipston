import requests
for i in range(1961, 2018):
    for j in range(1, 13):
        if j==2:
            if i%4==0:
                k = 29
            else:
                k = 28
        elif j==4 or j==6 or j==9 or j==11:
            k = 30
        else:
            k = 31
            
        if j<10:
            file ='chess-met_tas_gb_1km_daily_{}0{}01-{}0{}{}.nc'.format(i, j, i, j, k)  
        else:
            file ='chess-met_tas_gb_1km_daily_{}{}01-{}{}{}.nc'.format(i, j, i, j, k)
        print(file)
        url = 'https://catalogue.ceh.ac.uk/datastore/eidchub/2ab15bf0-ad08-415c-ba64-831168be7293/tas/'
        r = requests.get(url+file, allow_redirects=True)
        
        open('/home/ira/google-drive/Cambridge/Michaelmas 2020/flood_data/chess-met/tas/'+file, 'wb').write(r.content)                                                                                                                                                        