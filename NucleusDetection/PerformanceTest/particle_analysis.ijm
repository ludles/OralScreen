inputFolder="/data2/jiahao/ISBI_2020/FCRNeval/PerformanceTest/PredMasks/"; //getDirectory("Choose input folder");
outputFolder="/data2/jiahao/ISBI_2020/FCRNeval/PerformanceTest/Results/"; //getDirectory("Choose output folder for the results");
list=getFileList(inputFolder);

run("Set Measurements...", "area centroid redirect=None decimal=3");

for(i=0; i<list.length; i++) {
	path=inputFolder+list[i];
	if(endsWith(path,".jpg")) open(path);
	showProgress(i, list.length); 
	if(nImages>=1) {
		setAutoThreshold("Default dark no-reset");
		//run("Threshold...");
		run("Analyze Particles...", "display clear"); //can be adjusted to "clear previous results"
		// selectWindow("Results");
		//The following two lines removes the file extension
		i_img_name = indexOf(list[i], "_");
		img_name = substring(list[i], 0, i_img_name);
		saveAs("Results",outputFolder + "Results_" + img_name + ".csv");
		close();
		}
	}
//saveAs("Results", "D:/Materials/Office/OralCells/FCRN/Results/Results.csv");