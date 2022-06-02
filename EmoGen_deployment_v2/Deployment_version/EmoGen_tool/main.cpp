//NOTE: will need to run it as this for now: MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330 ./run_tool.sh
// Old attempt waas based on this: https://lencerf.github.io/post/2019-09-21-save-the-opengl-rendering-to-image-file/
// To build "cd /media/emogen_all_shared/emoGen/EmoGen_deployment_v2/Deployment_version/EmoGen_tool/build"
// and run "cmake ../ && make"

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
#include <opencv2/core/optim.hpp>


// ASSIMP HEADERS
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/cimport.h>


#include <stdlib.h>
#include <fstream>
#include <functional>
#include <time.h>
#include <random>
#include <chrono>
#include <map>
#include <algorithm>

// OpenGL
#include <GL/glew.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include "common/shader.hpp"
#include "common/objloader.hpp"
#include "common/controls.hpp"
#include "common/text2D.hpp"
#include "common/texture.hpp"

#include <gtk/gtk.h>
#include "box_multiwidget.hpp"

#include "data.hpp"
#include "utility.hpp"

#include <thread>
#include <mutex>

#include <SOIL/SOIL.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

int fileWatcherChecker = -1;

int GenNr_counter = 0;
const int MaxGen = 10;

int numberOfvrtx;
int NumberOfBlendshapes;
utility helper;

cv::Mat_<double> allBlendShapeVertices3Nx1;
std::vector<int> current_selections;

bool load_blendshapes_speedy(int expected_number);
cv::Mat get_localisation_point_cloud(std::string filename);
void load_initialisation(std::vector<unsigned int> &indices, std::vector<glm::vec3> &vertices, std::vector<glm::vec2> &uvs,  std::vector<glm::vec3> &normals);
void generateNextGen();
void update_faces(std::vector<glm::vec3> &vertices,  std::vector<glm::vec3> &normals);
void crossBreed(int chosen_nr2, std::vector<double>& face_to_cross_breed);
void mutateBlendshapes(std::vector<double> &face_to_mutate, double minMR, double MR, std::vector<int>&which_blnd_nrs);
void save_result();
void save_result_custom(int face_nr, int id);
std::vector<double> randBlendshapes(double MR, std::vector<int>&which_blnd_nrs);
void SaveImage(cv::Mat &img, std::string filename);
void save_custom_session(std::vector<std::vector<double>> weights, int generation, std::string user_dir);


std::string BLENDSHAPE_DIRECTORY;
std::string USER_DIRECTORY;
std::string WEB_BASE_DIRECTORY("/home/deploy/Emogen-Rails/shared/public/");
std::string orderOfblendshapes_FILE;
std::string Neutral_FILE;
std::string SHADER_DIRECTORY;
std::string EMOTION_TYPE;
std::string TARGET_EMOTION;
std::string SELECTION_INSTRUCTIONS;
std::string OUTPUT_DIRECTORY;
std::string INITIALISATION_FILE;
std::string NEUTRAL_IN_UPDATED_POSITION_FILE;

int maximum_number_of_generations;

EGLDisplay eglDisplay;
EGLSurface eglSurface;
EGLContext eglContext;

GLuint framebufferName;
GLuint renderTexture;
GLuint depthrenderbuffer;
GLFWwindow* window; //TODO: Need to remove it in the future

// default rendering setting
int window_width;  // = 1024;
int window_height; // = 768;
glm::vec3 camera_position = glm::vec3(0.0f, 176.0f, 950.0f);
glm::vec3 lightPos        = glm::vec3(0.0f, 176.0f, 950.0f);
float lookat_z = 11.0f;
float FOV = 5.0f;
float Znear = 910.0f;    // 0.1f;
float Zfar =  1010.0f;    // 1500.0f;

data * my_data;

std::vector<bool> active_faces;
std::vector<bool> monster_faces;
int eliteFace;
std::string eliteFace_save_name;
int exit_code = -1;
bool stall_interface = false;

cv::Mat translation;
cv::Mat rotation;
double scale;
bool doScale = false;


cv::Mat record;

std::ofstream monster_log;
std::string monster_log_filename;

//std::ofstream check_distribution;

std::vector<int> save_numbers;

bool random_initialisation;
int init_number;

bool protocol_generated_initialisation;
bool reinit_after_reset;


bool unit_test = false;

std::vector<int> sample_list;

std::mt19937 generator_global;

bool include_head_motion;
bool include_eye_pupil_motion;
bool include_eye_lid_motion;

int min_num_of_sel, max_num_of_sel;
bool continuing = false;

int main(int argc, char** argv) {

    active_faces.resize(10);
    monster_faces.resize(10);

    std::stringstream is_user_continuing(argv[27]);
    is_user_continuing >> std::boolalpha >> continuing; //conversion to bool
    GenNr_counter = atoi(argv[28]);
    USER_DIRECTORY = argv[50];

    window_width = atoi(argv[25]) * 0.5; //1024
    window_height = atoi(argv[26]) * (2.0/3.0); //768

    min_num_of_sel = atoi(argv[22]);
    max_num_of_sel = atoi(argv[23]);

    if ((max_num_of_sel < min_num_of_sel)
        || max_num_of_sel <= 0  || min_num_of_sel <= 0
        || max_num_of_sel > 10 ||  min_num_of_sel > 10 )  {

        return 1;
    }


    unsigned seed_global = std::chrono::system_clock::now().time_since_epoch().count();
    generator_global.seed(seed_global);

    std::stringstream protocol_generation(argv[20]);
    protocol_generation >> std::boolalpha >>  protocol_generated_initialisation;

    if (!protocol_generated_initialisation) {

        std::stringstream ss_init_choice(argv[15]);
        ss_init_choice >> std::boolalpha >>  random_initialisation;
        if(!random_initialisation) init_number = atoi(argv[16]);

    } else {

        std::stringstream ss_reset_strategy(argv[21]);
        ss_reset_strategy >> std::boolalpha >>  reinit_after_reset;
    }

    // blendshape choice restrictions
    std::stringstream ss_head_motion_choice(argv[17]);
    ss_head_motion_choice >> std::boolalpha >>  include_head_motion;

    std::stringstream ss_eye_pupil_motion_choice(argv[18]);
    ss_eye_pupil_motion_choice >> std::boolalpha >>  include_eye_pupil_motion;

    std::stringstream ss_eye_lid_motion_choice(argv[19]);
    ss_eye_lid_motion_choice >> std::boolalpha >>  include_eye_lid_motion;


    helper = utility();

    BLENDSHAPE_DIRECTORY=argv[1];
    orderOfblendshapes_FILE=argv[2];
    Neutral_FILE=argv[3];
    SHADER_DIRECTORY=argv[4];
    OUTPUT_DIRECTORY = argv[6];

    EMOTION_TYPE = argv[5];
    TARGET_EMOTION = EMOTION_TYPE.c_str();

    std::string delimiter = "_";

    size_t pos = TARGET_EMOTION.find(delimiter);
    if (pos == std::string::npos) TARGET_EMOTION = "";
    else TARGET_EMOTION = TARGET_EMOTION.erase(0, pos + delimiter.length());


    TARGET_EMOTION = "<b> Target expression: " + TARGET_EMOTION + " </b>";


    if ( min_num_of_sel == max_num_of_sel) {

        SELECTION_INSTRUCTIONS = "<b> Please select "  + std::to_string(min_num_of_sel) + " faces</b>";

    } else{

        SELECTION_INSTRUCTIONS = "<b> Please select min." + std::to_string(min_num_of_sel) + " and max. " + std::to_string(max_num_of_sel) + " faces </b>";
    }

    std::cout << "maximum_number_of_generations 0" << std::endl;
    maximum_number_of_generations=atoi(argv[7]);
    std::cout << "maximum_number_of_generations 1" << std::endl;

    const char * TEXTURE_FILE = argv[10];
    std::string texture_file_as_string(TEXTURE_FILE);
    std::string format = texture_file_as_string.substr( texture_file_as_string.length() - 3 );
    if (format != "png" && format!= "bmp") {
        return 1;
    }

    // 1a. initialise data structures
    my_data = new data();

    // 1b. get transform to default rendering position if neccessary

    // std::string target_barycentrics_filename=argv[8];
    // cv::Mat points = get_localisation_point_cloud(target_barycentrics_filename);
    // points.copyTo( my_data -> incoming_point_cloud );
    // helper.procrustes(my_data->default_point_cloud, my_data->incoming_point_cloud);

    // 1b. read transform from file
    cv::Mat info(4, 3, CV_64F);
    info.setTo(0.0);
    std::string info_filename =  BLENDSHAPE_DIRECTORY + "data2.dat";
    std::ifstream rf (info_filename, std::ios::in | std::ios::binary);
    if(!rf) {
        return 1;
    }
    rf.read(reinterpret_cast<char*> (info.data), 12 * sizeof(double));
    info.rowRange(0,3).copyTo(rotation);
    info.rowRange(3,4).copyTo(translation);
    rf.close();
    scale = 1.0;


    INITIALISATION_FILE = argv[9];
    NEUTRAL_IN_UPDATED_POSITION_FILE = argv[11];


    my_data -> collision_anchor_coordinates = helper.read_barycentrics(argv[12]);
    my_data -> collision_anchor_coordinates_teeth = helper.read_barycentrics(argv[13]);
    my_data -> lower_lip_coordinates = helper.read_barycentrics(argv[14]);

    auto start = std::chrono::steady_clock::now();
    bool check_model = load_blendshapes_speedy(atoi(argv[24]));
    auto end = std::chrono::steady_clock::now();

    if (!check_model) return 1;

    for (int nr = 0; nr < NumberOfBlendshapes; ++nr) {

        bool is_corrective = false;
        for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it)
            if (it -> second == nr)  { is_corrective = true; break; }
        if(is_corrective) continue;

        if (!include_head_motion && std::find(my_data -> head_motion_blnds.begin(), my_data -> head_motion_blnds.end(), nr) != my_data -> head_motion_blnds.end() ) continue;
        if (!include_eye_pupil_motion
            && std::find(my_data -> eye_pupil_motion_blnds.begin(), my_data -> eye_pupil_motion_blnds.end(), nr) != my_data -> eye_pupil_motion_blnds.end() ) continue;
        if (!include_eye_lid_motion
            && std::find(my_data -> eye_lid_motion_blnds.begin(), my_data -> eye_lid_motion_blnds.end(), nr) != my_data -> eye_lid_motion_blnds.end() ) continue;


        sample_list.push_back(nr);

    }


    int failed = my_data -> initialise(); //NOTE: here previous weights and current weights are set fir the first time alex
    if (failed == 1) return 1;


    monster_log_filename = OUTPUT_DIRECTORY + "monster_log.txt";
    std::ifstream check_monster_log(monster_log_filename);
    if(check_monster_log){
        check_monster_log.close();
        monster_log.open(monster_log_filename, std::ofstream::out | std::ofstream::app);

    } else {
        check_monster_log.close();
        monster_log.open(monster_log_filename, std::ofstream::out | std::ofstream::trunc);

    }
    monster_log.close();



    // 3. load 10 initialisation faces or generate 10 initialisation faces - DONE
    // Read our .obj file
    // OPENGL RENDERER
    std::vector<unsigned int> indices;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;

    start = std::chrono::steady_clock::now();
    load_initialisation(indices, vertices, uvs, normals);
    end = std::chrono::steady_clock::now();

    // to get the correctives into the previous initialisation
    my_data ->prepare_for_next_generation();

    // 3. visualise 10 faces + run GUI for clicking
    // initialise the EGL renderer

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

    glewExperimental=true;
    if (glewInit() != GLEW_OK) {
        return -1;
    }

    glClearColor(0.0f,0.0f, 0.0f, 1.0f);

    glEnable(GL_DEPTH_TEST); // Enable depth test
    glDepthFunc(GL_LESS);   // Accept fragment if it is closer to the camera than the former one

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    GLuint Texture;
    unsigned char* image;
    int tex_width, tex_height;
    if (format == "bmp") {
        Texture = loadBMP_custom(TEXTURE_FILE);
    } else if (format == "png") {
        image = SOIL_load_image(TEXTURE_FILE, &tex_width, &tex_height, 0, SOIL_LOAD_RGB);
    }

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

    std::string session_output_filename;

    if ( continuing ) {

        active_faces[0] = atoi(argv[29]);
        active_faces[1] = atoi(argv[30]);
        active_faces[2] = atoi(argv[31]);
        active_faces[3] = atoi(argv[32]);
        active_faces[4] = atoi(argv[33]);
        active_faces[5] = atoi(argv[34]);
        active_faces[6] = atoi(argv[35]);
        active_faces[7] = atoi(argv[36]);
        active_faces[8] = atoi(argv[37]);
        active_faces[9] = atoi(argv[38]);
        eliteFace = atoi(argv[39]);
        monster_faces[0] = atoi(argv[40]);
        monster_faces[1] = atoi(argv[41]);
        monster_faces[2] = atoi(argv[42]);
        monster_faces[3] = atoi(argv[43]);
        monster_faces[4] = atoi(argv[44]);
        monster_faces[5] = atoi(argv[45]);
        monster_faces[6] = atoi(argv[46]);
        monster_faces[7] = atoi(argv[47]);
        monster_faces[8] = atoi(argv[48]);
        monster_faces[9] = atoi(argv[49]);

        // record is zeroed in GenerateNextGeneration()
        // update_faces() creates populates global record with a header of 2 rows of length 10
        // and current weights of the #Blendshapes x 10 faces after application of correctives etc.

        record.at<double>(0,eliteFace - 1) = 1.0;
        for (int face_iter = 0; face_iter < active_faces.size(); ++face_iter)
            if ( active_faces[face_iter]!= false ) record.at<double>(1, face_iter) = 1.0;

        my_data -> full_account.push_back(record);
        session_output_filename = OUTPUT_DIRECTORY + EMOTION_TYPE + "_output_" + std::to_string(my_data->session_nr) +".csv";
        helper.write_session_to_csv_file(session_output_filename);


        my_data->chosen_rows.clear();
        my_data->chosen_rows.push_back((eliteFace - 1));
        for (int face_iter = 0; face_iter < active_faces.size(); ++face_iter)
            if ( active_faces[face_iter]!= false && face_iter != (eliteFace - 1) ) my_data->chosen_rows.push_back(face_iter);


        if (!(std::find(monster_faces.begin(), monster_faces.end(), true) == monster_faces.end())) {

            monster_log.open(monster_log_filename, std::ofstream::out | std::ofstream::app);
            time_t now = time(0);
            monster_log << ctime(&now);
            for (int face_iter = 0; face_iter < monster_faces.size(); ++face_iter) {
                if( monster_faces[face_iter] ) {
                    for(int i = 0; i < NumberOfBlendshapes; ++i)
                        monster_log << my_data->weights_current_generation[face_iter][i] << " ";
                    monster_log << std::endl;
                }

            }
            monster_log << std::endl;
            monster_log.close();

        }

        //Note: At this point no pic is generated

        if (GenNr_counter == maximum_number_of_generations ){
            // save_result();
            my_data->chosen_rows.clear();
            exit_code = 3;
            stall_interface = false;

        } else{
            exit_code = -1;
        }

        //Then start generating next gen based on the user choices

        // Generate new iteration
        generateNextGen(); //NOTE: this will generate new weights too

        auto start = std::chrono::steady_clock::now();
        update_faces(vertices, normals);
        auto end = std::chrono::steady_clock::now();


        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, &vertices[0], vertices.size() * sizeof(glm::vec3));
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
        ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, &normals[0], normals.size() * sizeof(glm::vec3));
        glUnmapBuffer(GL_ARRAY_BUFFER);


        GenNr_counter++;

        my_data ->prepare_for_next_generation();

        save_custom_session(my_data->weights_current_generation, GenNr_counter, USER_DIRECTORY);

        my_data->chosen_rows.clear();
        stall_interface = false;

    }


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(programID);
    glm::mat4 ProjectionMatrix =  glm::perspective(glm::radians(FOV), (float) window_width / (float) window_height, Znear, Zfar);

    glm::mat4 ViewMatrix = glm::lookAt(
            glm::vec3(camera_position.x, camera_position.y, camera_position.z),
            glm::vec3(camera_position.x, camera_position.y, lookat_z),
            glm::vec3(0, 1 ,0)
    );


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
            (void*)0
    );


    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            0,
            (void*)0
    );

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glVertexAttribPointer(
            2,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            (void*)0
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);



    initText2D((SHADER_DIRECTORY + "Holstein.DDS").c_str());
    printText2D("1", 115, 384, 50);
    printText2D("2", 299, 384, 50);
    printText2D("3", 485, 384, 50);
    printText2D("4", 670, 384, 50);
    printText2D("5", 860, 384, 50);
    printText2D("6", 102, 20, 50);
    printText2D("7", 299, 20, 50);
    printText2D("8", 485, 20, 50);
    printText2D("9", 670, 20, 50);
    printText2D("1", 840, 20, 50);
    printText2D("0", 873, 20, 50);

    cleanupText2D();

    glEnable(GL_DEPTH_TEST); // Enable depth test
    glDepthFunc(GL_LESS);   // Accept fragment if it is closer to the camera than the former one


    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    //glfwSwapBuffers(eglDisplay, eglSurface); //TODO: Enable this ?

    cv::Mat res(window_height, window_width, CV_8UC3, cv::Scalar(0, 0, 0));

    glReadPixels(0, 0, window_width, window_height, GL_BGR, GL_UNSIGNED_BYTE, res.data);
    cv::flip(res, res, 0);
    //return base64 to client
    SaveImage(res, WEB_BASE_DIRECTORY + USER_DIRECTORY + "/result.png");

    if (format == "png") SOIL_free_image_data(image);
    return 0;


}

bool load_blendshapes_speedy(int expected_number) {


    utility helper;

    Assimp::Importer importer;



    const aiScene * currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices  );
    aiMesh* mesh = currentScene->mMeshes[0];
    numberOfvrtx = mesh->mNumVertices;

    for (unsigned int k=0; k < numberOfvrtx; k++){

        cv::Mat ptt(1,3, CV_64F);
        ptt.at<double>(0) =  (double) mesh->mVertices[k].x;
        ptt.at<double>(1) =  (double) mesh->mVertices[k].y;
        ptt.at<double>(2) =  (double) mesh->mVertices[k].z;

        ptt = scale * ptt * rotation +  translation;

        mesh->mVertices[k].x = (float) ptt.at<double>(0);
        mesh->mVertices[k].y = (float) ptt.at<double>(1);
        mesh->mVertices[k].z = (float) ptt.at<double>(2);



    }

    cv::Mat collision_anchor_neutral = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates, false);
    cv::Mat collision_anchor_neutral_teeth = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates_teeth, false);
    cv::Mat lower_lip_anchor_neutral = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> lower_lip_coordinates, false);

    int counter = 0;

    Assimp::Exporter exporter;
    exporter.Export(currentScene, "obj",  NEUTRAL_IN_UPDATED_POSITION_FILE);
    Neutral_FILE = NEUTRAL_IN_UPDATED_POSITION_FILE;


    cv::Mat init(expected_number * numberOfvrtx * 3, 1, CV_64F);
    init.setTo(0.0);
    init.copyTo(allBlendShapeVertices3Nx1);
    std::string info_filename = BLENDSHAPE_DIRECTORY + "data.dat";
    std::ifstream rf (info_filename, std::ios::in | std::ios::binary);
    if(!rf) {
        return 1;
    }
    rf.read(reinterpret_cast<char*> (allBlendShapeVertices3Nx1.data), expected_number * numberOfvrtx * 3 * sizeof(double));
    rf.close();

    std::ifstream blnd_order_infi;
    blnd_order_infi.open(orderOfblendshapes_FILE);
    if(!blnd_order_infi.is_open()) {
        return false;
    }

    std::vector<std::string> blendshape_names;
    NumberOfBlendshapes = 0;


    std::map<std::string,int> lefts;
    std::map<std::string,int> rights;

    cv::Mat collision_anchor_blendshape(4 * my_data -> collision_anchor_coordinates.rows, 3, CV_64F);
    collision_anchor_blendshape.setTo(0.0);
    cv::Mat all_lwr_lip_anchor_blendshape(4 * my_data -> lower_lip_coordinates.rows, 3, CV_64F);
    all_lwr_lip_anchor_blendshape.setTo(0.0);
    cv::Mat collision_anchor_blendshape2(4 * my_data -> collision_anchor_coordinates.rows, 3, CV_64F);
    collision_anchor_blendshape2.setTo(0.0);

    // on teeth collision correctives
    cv::Mat collision_anchor_blendshape_teeth(4 * my_data -> collision_anchor_coordinates_teeth.rows, 3, CV_64F);
    collision_anchor_blendshape_teeth.setTo(0.0);
    cv::Mat all_lwr_lip_anchor_blendshape2(4 * my_data -> lower_lip_coordinates.rows, 3, CV_64F);
    all_lwr_lip_anchor_blendshape2.setTo(0.0);
    // on lip collision correctives
    cv::Mat collision_anchor_blendshape_teeth2(4 * my_data -> collision_anchor_coordinates_teeth.rows, 3, CV_64F);
    collision_anchor_blendshape_teeth2.setTo(0.0);

    while (!blnd_order_infi.eof()){

        std::string name;
        blnd_order_infi >> name;

        if (name == "") break;

        std::size_t found_special;
        int blnd_type = helper.left_or_right(name, found_special);

        if (blnd_type == 1) {
            std::string root = name.substr (found_special + 3);
            lefts[root] = NumberOfBlendshapes;
        }

        if (blnd_type == 2) {
            std::string root = name.substr (found_special + 3);
            rights[root] = NumberOfBlendshapes;
        }

        bool special_shape = helper.seals_and_collisions(name);
        if (special_shape) my_data -> correctives[name] = NumberOfBlendshapes;

        my_data -> blendshape_table[name] = NumberOfBlendshapes;

        blendshape_names.push_back(name);

        // LIP COLLISION CORRECTION DATA
        bool is_collision_blndsh = false;
        int collision_order_nr;
        for (std::map<int, std::string>::iterator it = my_data -> anchor_pair_nr_to_collision_blnd.begin();
             it != my_data->anchor_pair_nr_to_collision_blnd.end();
             ++it) {
            if (it -> second == name) {is_collision_blndsh = true; collision_order_nr = it -> first; break;}

        }

        if (is_collision_blndsh) {

            const aiScene * currentScene;
            currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices );
            aiMesh* mesh = currentScene->mMeshes[0];
            for (unsigned int vrtx_nr = 0; vrtx_nr < numberOfvrtx; vrtx_nr++){

                mesh->mVertices[vrtx_nr].x = mesh->mVertices[vrtx_nr].x + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx + 3 * vrtx_nr);
                mesh->mVertices[vrtx_nr].y = mesh->mVertices[vrtx_nr].y + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx + 3 * vrtx_nr + 1);
                mesh->mVertices[vrtx_nr].z = mesh->mVertices[vrtx_nr].z + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx + 3 * vrtx_nr + 2);
            }

            cv::Mat coordinates = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates, false);
            for (unsigned int collision_nr = 0; collision_nr < my_data -> collision_anchor_coordinates.rows; ++collision_nr){

                int order_nr = collision_order_nr * my_data -> collision_anchor_coordinates.rows + collision_nr;
                coordinates.row(collision_nr).copyTo(collision_anchor_blendshape.row(order_nr));

            }

            cv::Mat coordinates_teeth = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates_teeth, false);
            for (unsigned int collision_nr = 0; collision_nr < my_data -> collision_anchor_coordinates_teeth.rows; ++collision_nr){

                int order_nr = collision_order_nr * my_data -> collision_anchor_coordinates_teeth.rows + collision_nr;
                coordinates_teeth.row(collision_nr).copyTo(collision_anchor_blendshape_teeth2.row(order_nr));

            }

            cv::Mat coordinates_lwr_ptrs = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> lower_lip_coordinates, false);
            for (unsigned int lwr_lip_nr = 0; lwr_lip_nr < my_data -> lower_lip_coordinates.rows;++lwr_lip_nr){
                int order_nr = collision_order_nr * my_data -> lower_lip_coordinates.rows + lwr_lip_nr;
                coordinates_lwr_ptrs.row(lwr_lip_nr).copyTo(all_lwr_lip_anchor_blendshape.row(order_nr));
            }

        } else {

            for (std::map<int, std::string>::iterator it = my_data -> teeth_anchor_pair_nr_to_collision_blnd.begin();
                 it != my_data-> teeth_anchor_pair_nr_to_collision_blnd.end();
                 ++it) {
                if (it -> second == name) {is_collision_blndsh = true; collision_order_nr = it -> first; break;}

            }

            if (is_collision_blndsh) {


                const aiScene * currentScene;
                currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices );
                aiMesh* mesh = currentScene->mMeshes[0];
                for (unsigned int vrtx_nr = 0; vrtx_nr < numberOfvrtx; vrtx_nr++){

                    mesh->mVertices[vrtx_nr].x = mesh->mVertices[vrtx_nr].x + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx
                                                                                                        + 3 * vrtx_nr);
                    mesh->mVertices[vrtx_nr].y = mesh->mVertices[vrtx_nr].y + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx
                                                                                                        + 3 * vrtx_nr + 1);
                    mesh->mVertices[vrtx_nr].z = mesh->mVertices[vrtx_nr].z + allBlendShapeVertices3Nx1(NumberOfBlendshapes * 3 * numberOfvrtx
                                                                                                        + 3 * vrtx_nr + 2);
                }

                cv::Mat coordinates_teeth = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates_teeth, false);

                for (unsigned int collision_nr = 0; collision_nr < my_data -> collision_anchor_coordinates_teeth.rows; ++collision_nr){

                    int order_nr = collision_order_nr * my_data -> collision_anchor_coordinates_teeth.rows + collision_nr;
                    coordinates_teeth.row(collision_nr).copyTo(collision_anchor_blendshape_teeth.row(order_nr));

                }

                cv::Mat coordinates_upper = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> collision_anchor_coordinates, false);
                for (unsigned int collision_nr = 0; collision_nr < my_data -> collision_anchor_coordinates.rows; ++collision_nr){

                    int order_nr = collision_order_nr * my_data -> collision_anchor_coordinates.rows + collision_nr;
                    coordinates_upper.row(collision_nr).copyTo(collision_anchor_blendshape2.row(order_nr));

                }


                cv::Mat coordinates_lwr_ptrs = helper.get_anchor_point_cartersian_coordinates(mesh, my_data -> lower_lip_coordinates, false);
                for (unsigned int lwr_lip_nr = 0; lwr_lip_nr < my_data -> lower_lip_coordinates.rows;++lwr_lip_nr){
                    int order_nr = collision_order_nr * my_data -> lower_lip_coordinates.rows + lwr_lip_nr;
                    coordinates_lwr_ptrs.row(lwr_lip_nr).copyTo(all_lwr_lip_anchor_blendshape2.row(order_nr));
                }


            }

        }

        NumberOfBlendshapes++;

    }


    // sanity check
    if (expected_number != NumberOfBlendshapes) {

        return false;

    }

    for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it) {

        std::size_t found_corrective = (it -> first).find("_");
        std::vector<double> activation_vector(NumberOfBlendshapes, 0.0);

        if (found_corrective != std::string::npos) {
            char *name_ptr = new char[(it -> first).length() + 1];
            strcpy(name_ptr, (it -> first).c_str());
            char * core_blndsh_name = strtok (name_ptr,"_");

            while (core_blndsh_name != NULL){

                printf ("%s\n", core_blndsh_name);
                std::map<std::string, int>::iterator it_blendsh = my_data -> blendshape_table.find(std::string(core_blndsh_name));
                activation_vector[it_blendsh -> second] = 1.0;
                core_blndsh_name = strtok (NULL, "_");

            }

            my_data -> correctives_to_activation_vector[it -> second] = activation_vector;


        }
    }

    for (std::map<std::string,int>::iterator it_lft = lefts.begin(); it_lft!=lefts.end(); ++it_lft) {

        std::map<std::string, int>::iterator it_rgt = rights.find(it_lft->first);
        if (it_rgt != rights.end())  my_data ->left_right_pairs[it_lft -> second]= it_rgt -> second;
        else std::cout << "Error: left blendshape " << it_lft -> first << " does not have a corresponding right shape " << std::endl;

    }

    // LIP COLLISIONS

    cv::Mat NeutralToScale;
    collision_anchor_neutral.copyTo(NeutralToScale);
    for (int i = 0; i < (4 - 1); i++)
        cv::vconcat(NeutralToScale, collision_anchor_neutral, NeutralToScale);
    cv::Mat collision_deviation = collision_anchor_blendshape - NeutralToScale;

    collision_deviation.copyTo(my_data -> collision_deviations);

    collision_deviation = cv::Mat();

    collision_deviation = collision_anchor_blendshape2 - NeutralToScale;

    collision_deviation.copyTo(my_data -> collision_deviations2);

    collision_deviation = cv::Mat();

    NeutralToScale = cv::Mat();
    lower_lip_anchor_neutral.copyTo(NeutralToScale);
    for (int i = 0; i < (4 - 1); i++) cv::vconcat(NeutralToScale, lower_lip_anchor_neutral, NeutralToScale);

    collision_deviation =  all_lwr_lip_anchor_blendshape - NeutralToScale;
    collision_deviation.copyTo(my_data -> collision_deviations_lwr_lip);

    collision_deviation = cv::Mat();

    collision_deviation =  all_lwr_lip_anchor_blendshape2 - NeutralToScale;
    collision_deviation.copyTo(my_data -> collision_deviations_lwr_lip_teeth);

    collision_deviation = cv::Mat();

    // TEETH COLLISIONS
    NeutralToScale = cv::Mat();
    collision_anchor_neutral_teeth.copyTo(NeutralToScale);
    for(int i = 0; i < (4 - 1); i++)
        cv::vconcat(NeutralToScale, collision_anchor_neutral_teeth, NeutralToScale);

    collision_deviation = collision_anchor_blendshape_teeth - NeutralToScale;

    collision_deviation.copyTo(my_data -> collision_deviations_teeth);

    collision_deviation = cv::Mat();

    collision_deviation = collision_anchor_blendshape_teeth2 - NeutralToScale;

    collision_deviation.copyTo(my_data -> collision_deviations_teeth2);

    collision_deviation = cv::Mat();


    record = cv::Mat(NumberOfBlendshapes + 2, 10, CV_64F);
    record.setTo(0.0);


    return true;
}




void load_initialisation(std::vector<unsigned int> &indices, std::vector<glm::vec3> &vertices, std::vector<glm::vec2> &uvs,  std::vector<glm::vec3> &normals) {


    Assimp::Importer importer;
    for (int choice_nr = 0; choice_nr < 10 ; ++choice_nr) {

        int offset = vertices.size();

        const aiScene * currentScene = importer.ReadFile(Neutral_FILE, aiProcess_JoinIdenticalVertices);
        aiMesh* mesh = currentScene->mMeshes[0];

        bool puffs_eliminated = false;

        for(int i = 0; i < NumberOfBlendshapes; ++i) {

            if (my_data->weights_current_generation[choice_nr][i] == 0) continue;
            if (protocol_generated_initialisation) {
                if(std::find(my_data->head_motion_blnds.begin(), my_data->head_motion_blnds.end(), i) != my_data->head_motion_blnds.end()) continue;
                else if (std::find(my_data->puffs.begin(), my_data->puffs.end(), i) != my_data->puffs.end()) { puffs_eliminated = true; continue; }
            }

            for (unsigned int k=0; k < mesh->mNumVertices; k++){

                mesh->mVertices[k].x = mesh->mVertices[k].x + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k);
                mesh->mVertices[k].y = mesh->mVertices[k].y + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 1);
                mesh->mVertices[k].z = mesh->mVertices[k].z + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 2);

            }
        }

        // correctives and collisions only when the initialisation is protocol-generated
        if (protocol_generated_initialisation) {

            helper.apply_correctives(mesh, choice_nr);
            helper.compute_smooth_vertex_normals(mesh);

            bool lips, teeth;
            if (puffs_eliminated) {

                helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, true);
                if (lips) {

                    helper.apply_any_set_of_blendshapes(mesh, choice_nr, my_data -> puffs);
                    helper.compute_smooth_vertex_normals(mesh);
                    helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);


                } else if (!lips) {
                    for (unsigned int puff_blnd_nr = 0; puff_blnd_nr < my_data -> puffs.size(); ++puff_blnd_nr) {

                        int blnd_nr = my_data -> puffs[puff_blnd_nr];
                        if ( my_data->weights_current_generation[choice_nr][blnd_nr] != 0.0 ) {

                            my_data->weights_current_generation[choice_nr][blnd_nr] = 0.0;
                        }

                    }

                    helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);
                }


            } else helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);

            if (lips || teeth ) helper.apply_correctives(mesh, choice_nr);
            if (include_head_motion) helper.apply_head_motion(mesh, choice_nr);


            // update stored initialisation if initialisation is to be kept for after reset
            if (!reinit_after_reset) {

                for(int i = 0; i < NumberOfBlendshapes; ++i) {

                    if ( my_data -> initialisation.at<double>(choice_nr, i) != my_data->weights_current_generation[choice_nr][i] ) {

                        my_data -> initialisation.at<double>(choice_nr, i) = my_data->weights_current_generation[choice_nr][i];
                    }
                }
            }
        }

        helper.compute_smooth_vertex_normals(mesh);

        for(int i = 0; i < NumberOfBlendshapes; ++i) record.at<double>(i + 2, choice_nr) = my_data->weights_current_generation[choice_nr][i];


        for(unsigned int i=0; i<mesh->mNumVertices; i++) {

            aiVector3D pos = mesh->mVertices[i];
            // MOVE TO MY DATA
            if (choice_nr == 0) {
                pos.x = pos.x - 40.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 1) {
                pos.x = pos.x - 20.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 2) {
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 3) {
                pos.x = pos.x + 20.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 4) {
                pos.x = pos.x + 40.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 5) {
                pos.x = pos.x - 40.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 6) {
                pos.x = pos.x - 20.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 7) {
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 8) {
                pos.x = pos.x + 20.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 9) {
                pos.x = pos.x + 40.0;
                pos.y = pos.y - 20.0;
            }

            vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));

            aiVector3D UVW = mesh->mTextureCoords[0][i];
            uvs.push_back(glm::vec2(UVW.x, UVW.y));

            aiVector3D n = mesh->mNormals[i];
            normals.push_back(glm::vec3(n.x, n.y, n.z));



        }

        for (unsigned int i=0; i<mesh->mNumFaces; i++){

            indices.push_back(offset + mesh->mFaces[i].mIndices[0]);
            indices.push_back(offset + mesh->mFaces[i].mIndices[1]);
            indices.push_back(offset + mesh->mFaces[i].mIndices[2]);

        }

    }

}

void update_faces(std::vector<glm::vec3> &vertices,  std::vector<glm::vec3> &normals) {


    int numOfcollision_ptrs =  my_data -> collision_anchor_coordinates.rows;
    vertices.clear();
    normals.clear();

    Assimp::Importer importer;
    for (int choice_nr = 0; choice_nr < 10; ++choice_nr) {



        bool copied_exactly = std::find(my_data -> copied_exactly_IDs.begin(),
                                        my_data -> copied_exactly_IDs.end(), choice_nr) != my_data -> copied_exactly_IDs.end();


        const aiScene * currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices);
        aiMesh* mesh =  currentScene->mMeshes[0];

        bool puffs_eliminated = false;

        for(int i=0; i < NumberOfBlendshapes; ++i) {

            if (my_data->weights_current_generation[choice_nr][i] == 0 ) continue;
            else if(std::find(my_data->head_motion_blnds.begin(), my_data->head_motion_blnds.end(), i) != my_data->head_motion_blnds.end()) continue;
            else if (std::find(my_data->puffs.begin(), my_data->puffs.end(), i) != my_data->puffs.end()) { puffs_eliminated = true; continue; }

            bool is_corrective = false;
            for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it) {
                if (it -> second == i)  {
                    is_corrective = true;
                    break;
                }
            }


            // non-zero corrective is only allowed if there has been a session reset (exit_code == 2) or if this is one of the faces copied
            // should not happen but just in case any future developments inadvertently upset this..

            if(is_corrective && exit_code!=2 && !copied_exactly ) {
                my_data->weights_current_generation[choice_nr][i] = 0.0;
            };


            if (is_corrective && exit_code !=2 && !copied_exactly) continue;

            for (unsigned int k=0; k < mesh->mNumVertices; k++){

                mesh->mVertices[k].x = mesh->mVertices[k].x + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k);
                mesh->mVertices[k].y = mesh->mVertices[k].y + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 1);
                mesh->mVertices[k].z = mesh->mVertices[k].z + my_data->weights_current_generation[choice_nr][i]
                                                              * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 2);

            }
        }

        //  (protocol_generated_initialisation && reinit_after_reset && !copied_exactly) - always apply correctives, even after initialisation
        //  (except when copied exactly)
        //  (exit_code != 2 && !copied_exactly) - if not session reset and if the face is not copied exactly


        if ( (protocol_generated_initialisation && reinit_after_reset && !copied_exactly) || (exit_code != 2 && !copied_exactly) ) {

            helper.apply_correctives(mesh, choice_nr);
            helper.compute_smooth_vertex_normals(mesh);

            bool lips, teeth;
            if (puffs_eliminated) {

                helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, true);
                if (lips) {

                    helper.apply_any_set_of_blendshapes(mesh, choice_nr, my_data -> puffs);
                    helper.compute_smooth_vertex_normals(mesh);
                    helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);


                } else if (!lips) {
                    for (unsigned int puff_blnd_nr = 0; puff_blnd_nr < my_data -> puffs.size(); ++puff_blnd_nr) {

                        int blnd_nr = my_data -> puffs[puff_blnd_nr];
                        if ( my_data->weights_current_generation[choice_nr][blnd_nr] != 0.0 ) {

                            my_data->weights_current_generation[choice_nr][blnd_nr] = 0.0;
                        }

                    }

                    helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);
                }


            } else helper.correct_lip_and_teeth_collisions(mesh, choice_nr, lips, teeth, false);

            if (lips || teeth ) helper.apply_correctives(mesh, choice_nr);

        } else if (puffs_eliminated) {

            helper.apply_any_set_of_blendshapes(mesh, choice_nr, my_data -> puffs);

        }

        if (include_head_motion) helper.apply_head_motion(mesh, choice_nr);
        helper.compute_smooth_vertex_normals(mesh);



        // ATTENTION: IF ANY CORRECTIVE INVOLVES A PUFF SHAPE NEED TO AN ADDITIONAL APPLY CORRECTIVE RUN!

        cv::Mat data(10, NumberOfBlendshapes, CV_64F);
        data.setTo(0.0);
        for (int row_nr = 0; row_nr < 10; ++row_nr) {
            double* data_ptr = data.ptr<double>(row_nr,0);
            memcpy(data_ptr,
                   &my_data -> weights_current_generation[row_nr][0],
                   NumberOfBlendshapes * sizeof(double));
        }
        cv::transpose(data,data);
        cv::Mat header(2,10,CV_64F);
        header.setTo(0.0);
        cv::vconcat(header, data, data);
        data.copyTo(record);



        for(unsigned int i=0; i<mesh->mNumVertices; i++) {

            aiVector3D pos = mesh->mVertices[i];

            // MOVE TO MY DATA - offsets
            if (choice_nr == 0) {
                pos.x = pos.x - 40.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 1) {
                pos.x = pos.x - 20.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 2) {
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 3) {
                pos.x = pos.x + 20.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 4) {
                pos.x = pos.x + 40.0;
                pos.y = pos.y + 20.0;
            }

            if (choice_nr == 5) {
                pos.x = pos.x - 40.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 6) {
                pos.x = pos.x - 20.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 7) {
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 8) {
                pos.x = pos.x + 20.0;
                pos.y = pos.y - 20.0;
            }

            if (choice_nr == 9) {
                pos.x = pos.x + 40.0;
                pos.y = pos.y - 20.0;
            }

            vertices.push_back(glm::vec3(pos.x, pos.y, pos.z));

            aiVector3D n = mesh->mNormals[i];
            normals.push_back(glm::vec3(n.x, n.y, n.z));


        }



    }



}


void generateNextGen() {


    my_data -> copied_exactly_IDs.clear();


    unsigned seed_1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator_1(seed_1);
    std::uniform_int_distribution<int> distribution_1(0, (my_data->chosen_rows.size() - 1)); // 1 0 0 1 0 0 1 1 1 1

    unsigned seed_2 = seed_1;
    while(seed_2 == seed_1) seed_2 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator_2(seed_2);
    std::uniform_int_distribution<int> distribution_2(0, (sample_list.size() - 1));


    record.setTo(0.0);

    std::vector<int> order_of_presentation;
    for (int i=0; i<10; ++i) order_of_presentation.push_back(i);

    unsigned seed_3 = seed_2;
    while(seed_3 == seed_2) seed_3 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator_3(seed_3);
    std::shuffle (order_of_presentation.begin(), order_of_presentation.end(), generator_3);




    for (int face_nr = 0; face_nr < 10; ++face_nr) {


        if (face_nr == 0) { // pick the elite face

            memcpy(&my_data->weights_current_generation[order_of_presentation[0]][0],
                   &my_data->weights_previous_generation[my_data->chosen_rows[0]][0],
                   NumberOfBlendshapes * sizeof(double));

            my_data -> copied_exactly_IDs.push_back(order_of_presentation[0]);


        }  else if (face_nr == 1 && my_data->chosen_rows.size() > 2) {
            // Average selected faces; Note to self: maybe identical to next case if only 2 chosen faces; CORRECTED, SEND TO MUTATE INSTEAD

            std::vector<double> average;
            average.resize(NumberOfBlendshapes);
            memset(&average[0], 0, average.size() * sizeof(double));
            for (int chosen_nr=0; chosen_nr < my_data->chosen_rows.size(); ++chosen_nr) {

                for (int blnd_nr=0; blnd_nr < NumberOfBlendshapes; ++blnd_nr) {

                    // zero out all collision/seal weights
                    bool is_corrective = false;
                    for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it)
                        if (it -> second == blnd_nr)  { is_corrective = true; break; }
                    if(is_corrective) continue;

                    average[blnd_nr] = average[blnd_nr] + my_data->weights_previous_generation[my_data->chosen_rows[chosen_nr]][blnd_nr];

                }
            }

            // enforce symmetry
            for (std::map<int, int>::iterator it = my_data->left_right_pairs.begin(); it != my_data->left_right_pairs.end(); ++it) {
                if (average[it ->first] != average[it -> second]){

                    double lft_rgt_average = 0.5 * ( average[it ->first] + average[it -> second] );

                    average[it ->first] = lft_rgt_average;
                    average[it ->second] = lft_rgt_average;

                }

            }

            std::transform(average.begin(), average.end(), average.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1,
                                                                                      1.0 / ( my_data->chosen_rows.size()) ));

            memcpy(&my_data->weights_current_generation[order_of_presentation[1]][0],
                   &average[0],
                   NumberOfBlendshapes * sizeof(double));


        } else if (face_nr == 2 && GenNr_counter!=1 && GenNr_counter!=2) {
            // Average of elite face and another selected face; Note to self: why always the first non-elite? CORRECTED BELOW

            int chosen_nr;

            std::vector<double> average;
            average.resize(NumberOfBlendshapes);
            memset(&average[0], 0, average.size() * sizeof(double));

            if ( my_data->chosen_rows.size() < 2) {
                // NEW: if nothing to average with, copy the shape at the position in the previous set

                memcpy(&my_data->weights_current_generation[order_of_presentation[2]][0],
                       &my_data->weights_previous_generation[order_of_presentation[2]][0],
                       NumberOfBlendshapes * sizeof(double));



                my_data -> copied_exactly_IDs.push_back(order_of_presentation[2]);

                continue;

            }  else {

                chosen_nr = distribution_1(generator_1);
                while(chosen_nr == 0) chosen_nr = distribution_1(generator_1);

            }

            for (int blnd_nr=0; blnd_nr < NumberOfBlendshapes; ++blnd_nr)  {

                // zero out collision/seal blendshapes
                bool is_corrective = false;
                for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it)
                    if (it -> second == blnd_nr)  { is_corrective = true; break; }
                if(is_corrective) continue;

                average[blnd_nr] = average[blnd_nr] + my_data->weights_previous_generation[my_data->chosen_rows[0]][blnd_nr]
                                   + my_data->weights_previous_generation[my_data->chosen_rows[chosen_nr]][blnd_nr];

            }

            // enforce symmetry
            for (std::map<int, int>::iterator it = my_data->left_right_pairs.begin(); it != my_data->left_right_pairs.end(); ++it) {

                if (average[it ->first] != average[it -> second]){

                    double lft_rgt_average = 0.5 * ( average[it ->first] + average[it -> second] );

                    average[it ->first] = lft_rgt_average;
                    average[it ->second] = lft_rgt_average;

                }

            }

            std::transform(average.begin(), average.end(), average.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, 0.5 ));

            memcpy(&my_data->weights_current_generation[order_of_presentation[2]][0],
                   &average[0],
                   NumberOfBlendshapes * sizeof(double));




        } else if (face_nr < 6) { // mutation and cross-breeding


            int chosen_nr1  = distribution_1(generator_1);
            int chosen_nr2 = chosen_nr1;
            if (my_data->chosen_rows.size() >= 2){
                while(chosen_nr2 == chosen_nr1) chosen_nr2 = distribution_1(generator_1);
            }

            std::vector<double> new_face;
            new_face.resize(NumberOfBlendshapes);
            memcpy(&new_face[0], &my_data->weights_previous_generation[my_data->chosen_rows[chosen_nr1]][0], new_face.size() * sizeof(double));

            crossBreed(chosen_nr2, new_face);

            int numberOfchanges = 2;

            std::vector<int> which_blendshapes;

            for (int times = 0; times < numberOfchanges; ++times) which_blendshapes.push_back(sample_list[distribution_2(generator_2)]);


            mutateBlendshapes(new_face, 0.0, 1.0, which_blendshapes);

            // enfore symmetry
            for (unsigned int wch_bld_nr = 0; wch_bld_nr < which_blendshapes.size(); ++wch_bld_nr) {
                for (std::map<int, int>::iterator it = my_data->left_right_pairs.begin(); it != my_data->left_right_pairs.end(); ++it) {

                    if ( it->first == which_blendshapes[wch_bld_nr] ) {

                        new_face[it->second] = new_face[which_blendshapes[wch_bld_nr]];

                        break;

                    } else if (it->second == which_blendshapes[wch_bld_nr] ) {

                        new_face[it->first] = new_face[which_blendshapes[wch_bld_nr]];

                        break;
                    }
                }
            }

            memcpy(&my_data->weights_current_generation[order_of_presentation[face_nr]][0],
                   &new_face[0],
                   NumberOfBlendshapes * sizeof(double));


        } else { // completely new

            int numberOfchanges = 6;

            std::vector<int> which_blendshapes;

            for (int times = 0; times < numberOfchanges; ++times) which_blendshapes.push_back(sample_list[distribution_2(generator_2)]);

            std::vector<double> random_face = randBlendshapes(1.0, which_blendshapes);

            // enforce symmetry
            for (unsigned int wch_bld_nr = 0; wch_bld_nr < which_blendshapes.size(); ++wch_bld_nr) {

                for (std::map<int, int>::iterator it = my_data->left_right_pairs.begin(); it != my_data->left_right_pairs.end(); ++it) {

                    if ( it->first == which_blendshapes[wch_bld_nr] ) {

                        random_face[it->second] = random_face[which_blendshapes[wch_bld_nr]];

                        break;

                    } else if (it->second == which_blendshapes[wch_bld_nr] ) {

                        random_face[it->first] = random_face[which_blendshapes[wch_bld_nr]];

                        break;

                    }

                }
            }


            memcpy(&my_data->weights_current_generation[order_of_presentation[face_nr]][0],
                   &random_face[0],
                   NumberOfBlendshapes * sizeof(double));




        }
    }


}


void crossBreed(int chosen_nr2, std::vector<double>&face_to_cross_breed){

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);


    for (int blnd_nr=0; blnd_nr < NumberOfBlendshapes; ++blnd_nr) {

        // zero out collision/seal blendshapes
        bool is_corrective = false;
        for (std::map<std::string, int>::iterator it = my_data->correctives.begin(); it != my_data->correctives.end(); ++it) {
            if (it -> second == blnd_nr)  {
                face_to_cross_breed[blnd_nr] = 0.0;
                is_corrective = true;
                break;
            }
        }
        if (is_corrective) {
            continue;
        }

        double flip = distribution(generator);


        if (flip < 0.5) {

            face_to_cross_breed[blnd_nr] = my_data->weights_previous_generation[my_data->chosen_rows[chosen_nr2]][blnd_nr];

            for (std::map<int, int>::iterator it = my_data->left_right_pairs.begin(); it != my_data->left_right_pairs.end(); ++it) {

                if ( ( it->first == blnd_nr ) && ( face_to_cross_breed[it->second] != face_to_cross_breed[blnd_nr] ) ) {

                    face_to_cross_breed[it->second] = face_to_cross_breed[blnd_nr];

                    break;

                } else if ( ( it->second == blnd_nr ) && ( face_to_cross_breed[it->first] != face_to_cross_breed[blnd_nr] ) ) {

                    face_to_cross_breed[it->first] = face_to_cross_breed[blnd_nr];

                    break;

                }

            }
        }
    }



}


void mutateBlendshapes(std::vector<double> &face_to_mutate, double minMR, double MR, std::vector<int>&which_blnd_nrs) {


    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(minMR, MR);

    for (int blnd_nr=0; blnd_nr < which_blnd_nrs.size(); ++blnd_nr) {

        face_to_mutate[which_blnd_nrs[blnd_nr]] =  distribution(generator);

    }





}



std::vector<double> randBlendshapes(double MR, std::vector<int>&which_blnd_nrs) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0, MR);


    std::vector<double> new_face;
    new_face.resize(NumberOfBlendshapes);
    memset(&new_face[0], 0, new_face.size() * sizeof(double));

    for (int blnd_nr=0; blnd_nr < which_blnd_nrs.size(); ++blnd_nr) {

        new_face[which_blnd_nrs[blnd_nr]] =  distribution(generator);

    }

    return new_face;

}



void save_result(){


    if (my_data-> chosen_rows.size() == 0)  return;

    std::string output_weights;
    if ( eliteFace_save_name.empty()) output_weights = OUTPUT_DIRECTORY + "MALE_" + EMOTION_TYPE + "_" + std::to_string(my_data->session_nr) + ".txt";
    else  output_weights = OUTPUT_DIRECTORY + eliteFace_save_name + ".txt";

    std::ofstream outfi(output_weights);
    if(!outfi.is_open()) {
        return;
    }

    Assimp::Importer importer;
    const aiScene * currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices);
    aiMesh* mesh = currentScene->mMeshes[0];

    for(int i = 0; i < NumberOfBlendshapes; ++i) {

        outfi << my_data->weights_current_generation[my_data-> chosen_rows[0]][i] << std::endl;

        for (unsigned int k = 0; k< mesh->mNumVertices; k++){

            mesh->mVertices[k].x = mesh->mVertices[k].x + my_data->weights_current_generation[my_data-> chosen_rows[0]][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k);
            mesh->mVertices[k].y = mesh->mVertices[k].y + my_data->weights_current_generation[my_data-> chosen_rows[0]][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 1);
            mesh->mVertices[k].z = mesh->mVertices[k].z + my_data->weights_current_generation[my_data-> chosen_rows[0]][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 2);

        }
    }

    outfi.close();

    std::string output_mesh;
    if ( eliteFace_save_name.empty()) output_mesh = OUTPUT_DIRECTORY + "MALE_" + EMOTION_TYPE +  "_" + std::to_string(my_data->session_nr) + ".obj";
    else  output_mesh  = OUTPUT_DIRECTORY + eliteFace_save_name + ".obj";
    Assimp::Exporter exporter;
    exporter.Export(currentScene, "obj",  output_mesh);


}



void save_result_custom(int face_nr, int id){


    std::string output_weights;
    output_weights = "/home/nadejda/home/nadejda/EmoGen_data/output/example_to_show_darren/unit_test/unit_test4/" + std::to_string(id) + "_after.txt";

    std::ofstream outfi(output_weights);
    if(!outfi.is_open()) {
        return;
    }

    Assimp::Importer importer;
    const aiScene * currentScene = importer.ReadFile(Neutral_FILE,  aiProcess_JoinIdenticalVertices);

    aiMesh* mesh = currentScene->mMeshes[0];
    for(int i = 0; i < NumberOfBlendshapes; ++i) {

        outfi << my_data->weights_current_generation[face_nr][i] << std::endl;

        for (unsigned int k = 0; k< mesh->mNumVertices; k++){

            mesh->mVertices[k].x = mesh->mVertices[k].x + my_data->weights_current_generation[face_nr][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k);
            mesh->mVertices[k].y = mesh->mVertices[k].y + my_data->weights_current_generation[face_nr][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 1);
            mesh->mVertices[k].z = mesh->mVertices[k].z + my_data->weights_current_generation[face_nr][i]
                                                          * allBlendShapeVertices3Nx1(i * 3 * numberOfvrtx + 3 * k + 2);

        }
    }

    outfi.close();

    Assimp::Exporter exporter;

    // attention: Export .obj recomputes the normals, normals preserved when exporting as .ply
    std::string output_mesh;
    output_mesh  = "/home/nadejda/home/nadejda/EmoGen_data/output/example_to_show_darren/unit_test/unit_test4/" + std::to_string(id) + ".obj";
    exporter.Export(currentScene, "obj",  output_mesh);


}

cv::Mat get_localisation_point_cloud(std::string filename) {


    Assimp::Importer importer;
    const aiScene * currentScene = importer.ReadFile(Neutral_FILE, aiProcess_JoinIdenticalVertices);
    aiMesh* mesh = currentScene->mMeshes[0];

    std::string barycentrics_file(filename);
    std::ifstream infi(barycentrics_file);
    if(!infi.is_open()) {

        return cv::Mat();
    }
    cv::Mat localisation_anchors;
    for (int lndm_nr = 0; lndm_nr < 12; ++lndm_nr) {

        int face_id;
        double coord_1, coord_2;
        infi >>  face_id;
        infi >>  coord_1;
        infi >>  coord_2;

        const aiFace& face = mesh->mFaces[face_id];

        cv::Mat face_vrtx_1, face_vrtx_2, face_vrtx_3;

        aiVector3D pos = mesh->mVertices[face.mIndices[0] ];
        face_vrtx_1 = (cv::Mat_<double>(3,1) << pos.x,  pos.y,  pos.z);

        pos = mesh->mVertices[face.mIndices[1] ];
        face_vrtx_2 = (cv::Mat_<double>(3,1) << pos.x,  pos.y,  pos.z);

        pos = mesh->mVertices[face.mIndices[2]];
        face_vrtx_3 = (cv::Mat_<double>(3, 1) << pos.x,  pos.y,  pos.z);


        cv::Mat ptt = coord_1 * face_vrtx_1 + coord_2 * face_vrtx_2 + (1 - coord_1 - coord_2) * face_vrtx_3;

        transpose(ptt, ptt);
        localisation_anchors.push_back(ptt);

    }

    return localisation_anchors;
}

cv::Mat read_csv(std::string input_csv){

    std::ifstream file(input_csv);
    int counter = 0;
    cv::Mat data_stream;
    std::string::size_type sz;
    while ( file.good() ) {

        std::string value;
        getline ( file, value, ',' );


        std::size_t pos = value.find('\n');

        if ( pos != std::string::npos) {

            std::string last_value_in_line = value.substr (0, pos);
            data_stream.push_back(std::stod(last_value_in_line, &sz));
            counter++;

            std::string first_value_in_line = value.substr(pos + 1);
            if (!first_value_in_line.empty()) {
                data_stream.push_back(std::stod(first_value_in_line, &sz));
                counter++;
            }


        } else  {
            data_stream.push_back(std::stod(value, &sz));
            counter++;
        }


    }

    return data_stream;

}

void SaveImage(cv::Mat &img, std::string filename)
{

    if( filename.find(".floatImg") != std::string::npos)
    {
        // this should be an uncompressed float image.
        std::ofstream outfi;
        outfi.open( filename, std::ios::out | std::ios::binary );

        unsigned magic,w,h,c;
        magic = 820830001;
        w = img.cols;
        h = img.rows;
        if( img.type() == CV_32FC1 )
        {
            c = 1;
        }
        else if( img.type() == CV_32FC3 )
        {
            c = 3;
        }
        else
        {
            throw std::runtime_error("SaveImage: Image had neither 1 nor 3 channels.");
        }
        outfi.write( (char*)&magic, sizeof(magic) );
        outfi.write( (char*)&w, sizeof(w) );
        outfi.write( (char*)&h, sizeof(h) );
        outfi.write( (char*)&c, sizeof(c) );

        outfi.write( (char*)img.data, h*w*c*sizeof(float) );
        return;
    }

    // todo... use Magick instead of opencv. I have reasons for that... umm...
    // honest. Probably mostly to do with OpenCV normalising things etc... maybe...
    if( img.type() == CV_32FC3 || img.type() == CV_32FC1 )
    {
        img *= 255;
    }
    cv::imwrite(filename, img);
}

void save_custom_session(std::vector<std::vector<double>> weights, int generation, std::string user_dir){

    int rows = 150;
    int columns = 10;


    std::ofstream myfile_gen;
    myfile_gen.open(WEB_BASE_DIRECTORY + user_dir + "/saved_generation.csv");
    myfile_gen << generation;
    myfile_gen.close();

    std::ofstream myfile;
    myfile.open(WEB_BASE_DIRECTORY + user_dir + "/initialisation.csv"); //NOTE: Overwriting, alex
    for(int i = 0; i < rows; i++){
        for(int k = 0; k < columns; k++) {
            if(k < 9) {
                myfile << weights[k][i];
                myfile << ",";
            }else{ //last character
                myfile << weights[k][i];
            }
        }
        myfile << "\n" ;
    }
    myfile.close();

}
