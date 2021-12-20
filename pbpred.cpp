#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include<vector>
#include<string>
#include<sys/stat.h>
// #include<typeid.h>
using namespace std;
using namespace cv::dnn; 

cv::Mat convertTo3Channels(const cv::Mat& binImg)
{
   cv:: Mat single_channel, grayMat;
    cv::cvtColor(binImg, grayMat,cv::COLOR_RGB2GRAY);
	vector<cv::Mat> channels;
    for (int i=0;i<1;i++)
    {
        channels.push_back(grayMat);
    }
    cv::merge(channels,single_channel);
	printf("row : %d, col : %d, channles : %d \n", single_channel.rows, single_channel.cols, single_channel.channels());
    return single_channel;
}

void argMax(const cv::Mat& res, float &score, int &index)
{
	float* pdata = (float*)res.data ; 
	for(int i=0;i < 10; i++)
	{
		if(score<pdata[i] )
		{
			score = pdata[i] ;
			index = i;
		}
	}
}
void Funct(string image_file)
{
	Net MNIST_Net  =  readNetFromTensorflow("mnist_model_1channel.pb");
    string outNode = "output_class_1/Softmax";
	string inputNode ="input_image_1/Conv2D";
	
	// get layername 
 	vector<string> layer_names = MNIST_Net.getLayerNames();
	// for (int i = 0; i < layer_names.size(); i++) {
	// 	int id = MNIST_Net.getLayerId(layer_names[i]);
	// 	auto layer = MNIST_Net.getLayer(id);
	// 	printf("layer id : %d, \t\ttype : %s, \t\tname : %s \n", id, layer->type.c_str(), layer->name.c_str());
    //     // cout<<" layer id :  "<< id<<'\t'<< "  type :"<<layer->type.c_str()
    //     // <<'\t'<< " name: "<<layer->name.c_str()<<endl;
	// }

	cv::Mat image = cv::imread(image_file);
	cv::Mat image2 = convertTo3Channels(image);
	// printf(" to one channle row : %d, col : %d , channles : %d \n", image2.rows, image2.cols, image2.channels());

	image2 = blobFromImage(image2,1.0, cv::Size(224,224), cv::Scalar(0,0,0));
	MNIST_Net.setInput(image2);
    cv::Mat outs;
	MNIST_Net.forward(outs,outNode);

	float score=-1;
	int index=-1;
	argMax(outs, score, index);
	printf(" %.3f ---- %d ", score, index);

}

// 使用类 加载一次模型， 多次进行调用
class pbModel
{
	public:
		pbModel(string pbfile,string oNode);
		void run(string fileName);
		cv::Mat imageProcess(const cv::Mat& binImg);
	private:
		string m_pbPath;
		string m_outNode;
		Net PBNet;
};

pbModel::pbModel(string pbfile,string oNode)
{
	m_pbPath = pbfile;
	m_outNode = oNode;
	PBNet =  readNetFromTensorflow(m_pbPath);
}
void pbModel::run(string fileName)
{
	int pos = fileName.rfind("/");
	string baseName = fileName.substr(pos+1 ,fileName.size()-1);

	cv::Mat image = cv::imread(fileName);
	cv::Mat image2 = this->imageProcess(image);

	image2 = blobFromImage(image2,1.0, cv::Size(224,224), cv::Scalar(0,0,0));
	this->PBNet.setInput(image2);
    cv::Mat outs;
	this->PBNet.forward(outs,this->m_outNode);

	float score=-1;
	int index=-1;
	argMax(outs, score, index);
	printf("file : %s , cat : %d , score : %.2f \n", baseName.c_str(), index, score);
	// cout<<"file : "<<baseName<< " cat : "<< index << " score : " <<score<<endl;

}
cv::Mat pbModel::imageProcess(const cv::Mat& binImg)
{
    cv:: Mat single_channel, grayMat;
	if(binImg.channels() ==3)
	{
		cv::cvtColor(binImg, grayMat,cv::COLOR_RGB2GRAY);
	}
	else
	{
		grayMat = binImg;
	}

	vector<cv::Mat> channels;
    for (int i=0;i<1;i++)
    {
        channels.push_back(grayMat);
    }
    cv::merge(channels,single_channel);
	// printf("row : %d, col : %d, channles : %d \n", single_channel.rows, single_channel.cols, single_channel.channels());
    return single_channel;
}

void fileGlob(string dir, vector<string> &nameLists, string item=".png")
{
    long hfile = 0;
    struct _finddata_t fileinfo;
    string p ;
    hfile = _findfirst(p.assign(dir).append("\\*" + item).c_str(), &fileinfo);
    if (hfile!=-1)
    {
        do
        {
            if(fileinfo.attrib & _A_SUBDIR)
            {}
            else
            {
				string temp = dir;
				nameLists.push_back(temp.append("/").append(fileinfo.name));
            }
        }while(_findnext(hfile, &fileinfo)== 0);
        _findclose(hfile);
    }
	
}
int main()
{
	string testDir = "D:/MyNAS/SynologyDrive/CPP/pb_cpp/test_images";

	vector<string> nameLists;
	fileGlob(testDir, nameLists);
	string pbfile = "mnist_model_1channel.pb";
	string outNode = "output_class_1/Softmax";
	
	pbModel pbRun(pbfile, outNode);
	for(int i=0;i<nameLists.size();i++)
	{
			pbRun.run(nameLists[i]);
	}
	string image_file = "D:/MyNAS/SynologyDrive/CPP/pb_cpp/test_images/48.png";
	// printf("%s", image_file.c_str());
	// Funct(image_file);
	return 0;
}