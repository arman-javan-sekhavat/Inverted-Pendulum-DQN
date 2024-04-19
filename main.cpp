#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <mlpack.hpp>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>



// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;



void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}



void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}


void mouse_move(GLFWwindow* window, double xpos, double ypos) {

    if (!button_left && !button_middle && !button_right) {
        return;
    }

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;


    int width, height;
    glfwGetWindowSize(window, &width, &height);


    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) == GLFW_PRESS;
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;


    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}



void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}



//****************************************************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//****************************************************

using namespace arma;
using namespace mlpack;

const double rand_max = (double) RAND_MAX;
const double F = 10;
const double alpha = 1.0;
const double gamma = 0.9;
double epsilon = 1.0;
const double time_step = m->opt.timestep;

double x = 0, x_dot = 0;
double theta = 0, theta_dot = 0;
double R = 0;
const double c1 = 1 - gamma;
const double c2 = 0.0;

const double x_max = 10;
const double theta_max = M_PI / 2.0;

unsigned short episode_num = 0;
size_t step_num = 0;
size_t r = 0;

const short N = 50;

short action = 0;

mat s(4, N, fill::zeros);
mat S(4, 1, fill::zeros);
mat q(2, N, fill::zeros);
mat q_hat(2, 1, fill::zeros);
mat old_q_hat(2, 1, fill::zeros);

FFN<MeanSquaredError, RandomInitialization>* p_model = nullptr;



void controller(const mjModel* m, mjData* d) {

    if (rand() / rand_max < epsilon) {
        action = (rand() / rand_max < 0.5) ? 0 : 1;
    }
    else {
        action = (q_hat(0, 0) > q_hat(1, 0)) ? 0 : 1;
    }

    d->ctrl[0] = (action == 0) ? (+F) : (-F);

}



// main function
int main(int argc, const char** argv) {
    
    if (argc != 2) {
        std::printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(argv[1]) > 4 && !std::strcmp(argv[1] + std::strlen(argv[1]) - 4, ".mjb")) {
        m = mj_loadModel(argv[1], 0);
    }
    else {
        m = mj_loadXML(argv[1], 0, error, 1000);
    }
    if (!m) {
        mju_error("Load model error: %s", error);
    }

    // make data
    d = mj_makeData(m);

    
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);


    // ********************************************************************************************
    // ********************************************************************************************
    // ********************************************************************************************
    // ********************************************************************************************

    FFN<MeanSquaredError, RandomInitialization> model(MeanSquaredError(), RandomInitialization(-1.0, 1.0));
    p_model = &model;

    model.Add<Linear>(4);
    model.Add<ReLU>();

    model.Add<Linear>(16);
    model.Add<ReLU>();

    model.Add<Linear>(64);
    model.Add<ReLU>();

    model.Add<Linear>(64);
    model.Add<ReLU>();

    model.Add<Linear>(16);
    model.Add<ReLU>();

    model.Add<Linear>(2);
    model.Add<Sigmoid>();

    ens::Adam optimizer(0.001, N, 0.9, 0.999, 1.0E-8, 40*N);


    mjcb_control = controller;


    while (!glfwWindowShouldClose(window)) {
        
        mjtNum simstart = d->time;


        while (d->time - simstart < 1.0 / 60.0) {

            x = d->qpos[0];
            theta = d->qpos[1];
            x_dot = d->qvel[0];
            theta_dot = d->qvel[1];

            r = step_num % N;

            s(0, r) = x;
            s(1, r) = x_dot;
            s(2, r) = theta;
            s(3, r) = theta_dot;

            S(0, 0) = x;
            S(1, 0) = x_dot;
            S(2, 0) = theta;
            S(3, 0) = theta_dot;

            if (r == (N - 1)) {
                model.Train(s, q, optimizer);
            }

            //***
            old_q_hat = q_hat;

            model.Predict(S, q_hat);
            R = (c1/2) * (1.5*exp(-fabs(5*theta)) + 0.5/cosh(3*x));

            if (step_num > 1) {
                q(action, (step_num - 1) % N) = old_q_hat(action, 0) + alpha * (R + gamma * q_hat.max() - old_q_hat(action, 0));
                q(1 - action, (step_num - 1) % N) = old_q_hat(1 - action, 0);
            }


            mj_step(m, d);
            step_num++;

        }

        if ((fabs(x) > x_max) || (fabs(theta) > theta_max)) {
            mj_resetData(m, d);
            episode_num++;

            epsilon -= 0.0005;

            std::cout << "Episode " << episode_num << std::endl;
            step_num = 0;
        }


    // ********************************************************************************************
    // ********************************************************************************************
    // ********************************************************************************************
    // ********************************************************************************************


        
        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        
        glfwSwapBuffers(window);

        
        glfwPollEvents();
    }

    
    mjv_freeScene(&scn);
    mjr_freeContext(&con);


    mj_deleteData(d);
    mj_deleteModel(m);

    
#if defined(__APPLE__)  defined(_WIN32)
    glfwTerminate();
#endif

    return 0;
}