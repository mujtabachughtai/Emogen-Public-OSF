file = File.open("/home/emogen_all/emoGen/EmoGen_deployment_v2/Deployment_version/web_data_temp.json")
content = file.read
file.close

open("/home/emogen_all/emoGen/EmoGen_deployment_v2/Deployment_version/web_data.json", "w") { |f|
	f.write content
}

