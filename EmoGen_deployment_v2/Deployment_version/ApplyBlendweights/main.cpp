
#include <math.h>
// opencv
# include <opencv2/highgui/highgui.hpp>
# include "opencv2/imgproc/imgproc.hpp"
# include <opencv2/core/core.hpp>
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv/cv.h"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/opencv.hpp"
# include  <opencv2/tracking/tracker.hpp>

// OpenGL
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <GL/gl.h>
#include "../EmoGen_tool/common/shader.hpp"
#include "../EmoGen_tool/common/texture.hpp"

// ASSIMP HEADERS
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#include <string> 
#include <stdlib.h>   

#include <SOIL/SOIL.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

cv::Mat_<double> allBlendShapeVertices3Nx1;  
std::string DATA_DIRECTORY;
std::string orderOfblendshapes_FILE;
std::string Neutral_FILE;
std::string weights_FILE;
std::string output_PATH;
std::string image_FILE;
std::string SHADER_DIRECTORY;
int NumberOfBlendshapes;

EGLDisplay eglDisplay;
EGLSurface eglSurface;
EGLContext eglContext;

GLFWwindow* window;

// default rendering setting
const int window_width         = 720; 
const int window_height        = 1280;
const int crop_corner_width    = 138;
const int crop_corner_height   = 249; 
const int cropped_image_width  = 420;  
const int cropped_image_height = 616; 
glm::vec3 camera_position = glm::vec3(0.0f, 176.0f, 950.0f);
glm::vec3 lightPos        = glm::vec3(0.0f, 176.0f, 950.0f); 
float lookat_z = 11.0f;
float FOV      = 5.0f;
float Znear    = 910.0f;   
float Zfar     = 1010.0f;    

void compute_smooth_vertex_normals(aiMesh* mesh);
void write_buffer_to_file(unsigned char pixels[cropped_image_height * cropped_image_width * 4 ]);

int main(int argc, char** argv) {

	DATA_DIRECTORY = argv[1];
	
	NumberOfBlendshapes = atoi(argv[2]);
	
	weights_FILE = argv[3];
	std::string weights(weights_FILE);
	std::ifstream infi_weights;
	char comma;
	std::string line;
	
	output_PATH = argv[4];
	
	const char * TEXTURE_FILE = argv[5];
	std::string texture_file_as_string(TEXTURE_FILE);
	std::string format = texture_file_as_string.substr( texture_file_as_string.length() - 3 );
	if (format != "png" && format!= "bmp") {
		return 1;
	}
	

	SHADER_DIRECTORY = argv[6];
	
	int total = atoi(argv[7]);
	
	std::stringstream ss_init_choice(argv[8]); 
	bool obj_file;
	ss_init_choice >> std::boolalpha >>  obj_file;
	
	
	cv::Mat info(4, 3, CV_64F);
	info.setTo(0.0);
	std::string info_filename = DATA_DIRECTORY + "data2.dat";
	std::ifstream rf (info_filename, std::ios::in | std::ios::binary);
	if(!rf) {
	  return 1;
	}
	rf.read(reinterpret_cast<char*> (info.data), 12 * sizeof(double));
	cv::Mat part1, part2;
	info.rowRange(0,3).copyTo(part1);
	info.rowRange(3,4).copyTo(part2);
	rf.close();
	
	// check rendering 
	Neutral_FILE = DATA_DIRECTORY + "neutral.obj";

	Assimp::Importer importer;
	const aiScene * currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices ); 
	if (currentScene == NULL) {
			return 1;
	}
	aiMesh* mesh = currentScene->mMeshes[0];
	
	int numberOfvrtx = mesh->mNumVertices; 

	cv::Mat init(NumberOfBlendshapes * numberOfvrtx * 3, 1, CV_64F);
	init.setTo(0.0);
	init.copyTo(allBlendShapeVertices3Nx1);
	std::string info_filename2 = DATA_DIRECTORY + "data.dat";
	std::ifstream rf2 (info_filename2, std::ios::in | std::ios::binary);
	if(!rf2) {
		return 1;
	}
	rf2.read(reinterpret_cast<char*> (allBlendShapeVertices3Nx1.data), NumberOfBlendshapes * numberOfvrtx * 3 * sizeof(double));
	rf2.close();

//	glewExperimental = true;
//	if( !glfwInit() ) {
//		return -1;
//	}

//	glfwWindowHint(GLFW_SAMPLES, 4);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
//	glfwWindowHint(GLFW_DECORATED, GL_FALSE);
//
//	window = glfwCreateWindow(window_width, window_height, "Input Face", NULL, NULL);
//
//	if( window == NULL ) {
//		glfwTerminate();
//		return -1;
//	}
//
//	glfwMakeContextCurrent(window);
//	if (glewInit() != GLEW_OK) {
//		return -1;
//	}
	
	unsigned char* image;
	int tex_width, tex_height;
	if (format == "png") image = SOIL_load_image(TEXTURE_FILE, &tex_width, &tex_height, 0, SOIL_LOAD_RGB);
	
	Assimp::Exporter exporter;
	infi_weights.open(weights);
	for (unsigned int sample_nr = 0; sample_nr < total; ++sample_nr) {

        //EGL code starts here - alex
        EGLDeviceEXT eglDevs[10];
        EGLint numDevices;

        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
        if( eglQueryDevicesEXT == NULL )
        {
            exit(0);
        }

        PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC) eglGetProcAddress("eglQueryDeviceStringEXT");
        PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC) eglGetProcAddress("eglQueryDeviceAttribEXT");

        eglQueryDevicesEXT(10, eglDevs, &numDevices);

        int dcToUse = 0;
        for( unsigned dc = 0; dc < numDevices; ++dc )
        {
            std::string devName( "N/A" );
            std::string devExtensions( eglQueryDeviceStringEXT( eglDevs[dc], EGL_EXTENSIONS ) );

            if( devExtensions.find("drm") != std::string::npos && dcToUse < 0)
            {
                dcToUse = dc;
            }
        }

        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");

        eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[dcToUse], 0);

        EGLint major, minor;
        eglInitialize(eglDisplay, &major, &minor);

        EGLint configAttribs[] =
                {
                        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                        EGL_BLUE_SIZE, 8,
                        EGL_GREEN_SIZE, 8,
                        EGL_RED_SIZE, 8,
                        EGL_DEPTH_SIZE, 24,
                        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                        EGL_NONE
                };

        EGLint numConfigs;
        EGLConfig eglCfg;
        eglChooseConfig(eglDisplay, configAttribs, &eglCfg, 1, &numConfigs);

        EGLint pbufferAttribs[] =
                {
                        EGL_WIDTH, window_width,
                        EGL_HEIGHT, window_height,
                        EGL_NONE
                };

        EGLSurface eglSurface = eglCreatePbufferSurface(eglDisplay, eglCfg, pbufferAttribs);
        if( eglSurface == EGL_NO_SURFACE )
        {
            exit(0);
        }

        eglBindAPI(EGL_OPENGL_API);

        EGLint ctxAttribs[] =
                {
                        EGL_CONTEXT_MAJOR_VERSION, 4,
                        EGL_CONTEXT_MINOR_VERSION, 1
                };

        eglContext = eglCreateContext(eglDisplay, eglCfg, EGL_NO_CONTEXT, NULL);
        if( eglContext == NULL )
        {
            exit(0);
        }

        eglSwapInterval( eglDisplay, 0 );

        eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        // we should be able to ask OpenGL about which version we've got...
        GLint glMajor, glMinor;
        glGetIntegerv(GL_MAJOR_VERSION, &glMajor);
        glGetIntegerv(GL_MINOR_VERSION, &glMinor);

        // EGL Code ends here - Alex
        std::cout << " Success " << std::endl;
        exit(0);

		cv::Mat_<double>  weights = cv::Mat(1, NumberOfBlendshapes, CV_64F, double(0.0));

		for(int i = 0; i < NumberOfBlendshapes; ++i) { 
			infi_weights >> weights(i); 
			infi_weights >> comma;
		}
		currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices );
		if (currentScene == NULL) {
			return 1;
		}
		mesh = currentScene->mMeshes[0];

		for (unsigned int k=0; k < numberOfvrtx; ++k){
			
			cv::Mat ptt(1,3, CV_64F);
			ptt.at<double>(0) =  (double) mesh->mVertices[k].x;
			ptt.at<double>(1) =  (double) mesh->mVertices[k].y;
			ptt.at<double>(2) =  (double) mesh->mVertices[k].z;
			
			ptt = ptt  * part1 + part2;
			
			mesh->mVertices[k].x = (float) ptt.at<double>(0);
			mesh->mVertices[k].y = (float) ptt.at<double>(1);
			mesh->mVertices[k].z = (float) ptt.at<double>(2);
			
		} 
		
		for(int i = 0; i < NumberOfBlendshapes; ++i) {
			for (unsigned int vrtx_nr = 0; vrtx_nr < numberOfvrtx; ++vrtx_nr){
				mesh->mVertices[vrtx_nr].x +=  weights(i) * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * vrtx_nr);
				mesh->mVertices[vrtx_nr].y +=  weights(i) * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * vrtx_nr + 1);
				mesh->mVertices[vrtx_nr].z +=  weights(i) * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * vrtx_nr + 2);
			}
		} 
		
		compute_smooth_vertex_normals(mesh);

		glClearColor(0.0f,0.0f, 0.0f, 1.0f);

		std::vector<unsigned int> indices;          
		std::vector<glm::vec3> vertices;
		std::vector<glm::vec2> uvs;
		std::vector<glm::vec3> normals; 
		
		for(unsigned int i=0; i<mesh->mNumVertices; ++i) {
		
			aiVector3D pos = mesh->mVertices[i]; 
			vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));

			aiVector3D UVW = mesh->mTextureCoords[0][i]; 
			uvs.push_back(glm::vec2(UVW.x, UVW.y));

			aiVector3D n = mesh->mNormals[i];
			normals.push_back(glm::vec3(n.x, n.y, n.z));

		}
		for (unsigned int i=0; i<mesh->mNumFaces; i++){
			
			indices.push_back(mesh->mFaces[i].mIndices[0]);
			indices.push_back(mesh->mFaces[i].mIndices[1]);
			indices.push_back(mesh->mFaces[i].mIndices[2]);
			
		} 
		
		GLuint VertexArrayID;
		glGenVertexArrays(1, &VertexArrayID);
		glBindVertexArray(VertexArrayID);
		
        GLuint Texture;
        if (format == "bmp") Texture = loadBMP_custom(TEXTURE_FILE);
		
		GLuint programID = LoadShaders( (SHADER_DIRECTORY + "StandardShading.vertexshader").c_str(),  (SHADER_DIRECTORY+"StandardShading.fragmentshader").c_str());
		GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

		// Get a handle for our "MVP" uniform
		GLuint MatrixID = glGetUniformLocation(programID, "MVP");
		GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
		GLuint ModelMatrixID = glGetUniformLocation(programID, "M");
		GLuint LightID = glGetUniformLocation(programID, "LightPosition_worldspace");

		GLuint vertexbuffer;
		glGenBuffers(1, &vertexbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_DYNAMIC_DRAW);

		GLuint uvbuffer;
		glGenBuffers(1, &uvbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_DYNAMIC_DRAW);

		GLuint normalbuffer;
		glGenBuffers(1, &normalbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_DYNAMIC_DRAW);

		GLuint elementbuffer;
		glGenBuffers(1, &elementbuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_DYNAMIC_DRAW);

		glEnable(GL_DEPTH_TEST); // Enable depth test
		glDepthFunc(GL_LESS);    // Accept fragment if it is closer to the camera than the former one

		char mesh_FILE_ext[100]; 
		sprintf(mesh_FILE_ext, "sample_%d.obj", sample_nr);
		std::string mesh_FILE = output_PATH + mesh_FILE_ext;
		char image_FILE_ext[100]; 
		sprintf(image_FILE_ext, "sample_%d.jpg", sample_nr);
		image_FILE = output_PATH + image_FILE_ext;
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(programID);
		glm::mat4 ProjectionMatrix =  glm::perspective(glm::radians(FOV), (float) window_width / (float) window_height, Znear, Zfar); 

		glm::mat4 ViewMatrix = glm::lookAt(
		glm::vec3(camera_position.x, camera_position.y, camera_position.z),   
		glm::vec3(camera_position.x, camera_position.y, lookat_z),   
		glm::vec3(0, 1 ,0) );
		 
		glm::mat4 ModelMatrix =  glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
		glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		if (format == "png") {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
		glUniform1i(TextureID, 0);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			   0,                  
			   3,                  
			   GL_FLOAT,           
			   GL_FALSE,          
			   0,                 
			   (void*)0       );
  
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                               
			2,                               
			GL_FLOAT,                         
			GL_FALSE,                         
			0,                                
			(void*)0         );

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glVertexAttribPointer(
				 2,                                
				 3,                                
				 GL_FLOAT,                        
				 GL_FALSE,                         
				 0,                               
				 (void*)0     );

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0); 

		unsigned char pixels[cropped_image_height * cropped_image_width  * 4 ] = { 0 };
		glReadPixels(crop_corner_width, crop_corner_height, cropped_image_width, cropped_image_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels );
		write_buffer_to_file(pixels); 
		  
		glDeleteBuffers(1, &vertexbuffer);
		glDeleteBuffers(1, &uvbuffer);
		glDeleteBuffers(1, &normalbuffer);
		glDeleteBuffers(1, &elementbuffer);
		glDeleteProgram(programID);
		glDeleteTextures(1, &Texture);
		glDeleteVertexArrays(1, &VertexArrayID);
		
		if (obj_file) exporter.Export(currentScene, "obj",  mesh_FILE);
	}
//	glfwDestroyWindow(window);
	if (format == "png") SOIL_free_image_data(image);
	return 0;

}

void compute_smooth_vertex_normals(aiMesh* mesh){

	// zero out all normals
	for (unsigned int vrtx_id = 0; vrtx_id < mesh->mNumVertices; ++vrtx_id) mesh->mNormals[vrtx_id] = aiVector3D(0.0f,0.0f,0.0f); 

	for (unsigned int face_id = 0; face_id < mesh->mNumFaces; ++face_id){
		
			const aiFace& face = mesh->mFaces[face_id]; 
			aiVector3D a = mesh->mVertices[face.mIndices[0]]; 
			aiVector3D b = mesh->mVertices[face.mIndices[1]]; 
			aiVector3D c = mesh->mVertices[face.mIndices[2]]; 

			aiVector3D e1  = a - b;
			aiVector3D e2  = c - b;
			aiVector3D normal = -(e1 ^ e2); 

			mesh->mNormals[face.mIndices[0]] += normal;
			mesh->mNormals[face.mIndices[1]] += normal;
			mesh->mNormals[face.mIndices[2]] += normal;
			
	}
	for (unsigned int vrtx_id = 0; vrtx_id < mesh->mNumVertices; ++vrtx_id) mesh->mNormals[vrtx_id].Normalize();
}


void write_buffer_to_file(unsigned char pixels[cropped_image_height * cropped_image_width * 4 ]){

	cv::Mat image(cropped_image_height, cropped_image_width, CV_8UC3);
	image.setTo(cv::Vec3b(0,0,0));
	
	int counter = 0;
	for (int row_nr = 0; row_nr < cropped_image_height; ++row_nr) {
		for (int col_nr = 0; col_nr < cropped_image_width ; ++col_nr) {

			 int blue  = counter * 4 + 2;
			 int green = counter * 4 + 1;
			 int red   = counter * 4;
			 
			 image.at<cv::Vec3b>(cropped_image_height - (row_nr + 1), col_nr)[0] = static_cast< int >(  pixels[ blue ] );
			 image.at<cv::Vec3b>(cropped_image_height - (row_nr + 1), col_nr)[1] = static_cast< int >(  pixels[ green ] );
			 image.at<cv::Vec3b>(cropped_image_height - (row_nr + 1), col_nr)[2] = static_cast< int >(  pixels[ red ] );

			 counter++;
		}
	}
	imwrite(image_FILE, image);
}




