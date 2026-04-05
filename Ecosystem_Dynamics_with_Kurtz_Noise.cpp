/* PREPROCESSOR INCLUSION */
//#include <cmath>  // Sono già contenuti in altri include ma per fortuna sono anche dotati di include guard
//#include <ctime>  // Forse si possono cancellare
//#include <cstring>
//#include <cstdio> 
//#include <cstdlib>
//#include <chrono>

#include "pietro_toolbox.h"
#include "kurtz_bath_eco_toolbox.h"

/* PREPROCESSOR DEFINITION */

#define SAVE_TO_BINARY
//#undef SAVE_TO_BINARY

/* STRUCTURE */

/* TYPE DEFINITION */

/* PROTOTYPES */

/* GLOBAL VARIABLES */

/* MAIN BEGINNING */

int main(int argc, char **argv){
    if(argc != 6) {
        fprintf(stderr,
                "Usage: %s N c p_A omega N_ext\n",
                argv[0]);
        return MY_FAIL;
    }
    clock_t begin = clock(); //Clock iniziale
    float time_spent;
    char *N_label = argv[1]; // Ricorda argv[0] è il nome dell'eseguibile!
    int N = atoi(N_label);
    char *c_label = argv[2];
    double c = atof(c_label);
    char *pA_label = argv[3];
    double p_A = atof(pA_label);
    char *omega_label = argv[4];
    double omega = atof(omega_label);
    int N_ext = atoi(argv[5]); 
    int N_meas = 1, N_previous_meas = 0;
    double lambda = 1e-6;

    double sigma = sqrt(1.0 / 4.0 / c);
    sigma = 1e-5;

    //void (*ia_generator)(double*, double*, double, double, double) = &partyally_asymmetric_gaussian;
    //void (*ia_generator)(double*, double*, double, double, double, double, double, double) = &asymmetric_antagonistic_lognormal_extraction;
    //char gr_label[] = "Trivially";
    //double (*gr_generator)(double, double) = &UNG; //&trivial_generator; // Pointer to a function (weird syntax but somehow it has a reason)
    //double gr_value = 1.0; 
    //double growth_rate_interval[2] = {1.0, 1.0};
    //char cc_label[] = "Trivially";
    //double (*cc_generator)(double, double) = &UNG; //&trivial_generator; // Pointer to a function (weird syntax but somehow it has a reason)
    //double carrying_cap_interval[2] = {1.0, 1.0};
    //double cc_value = 1.0;

    double prob_link = c / (double) (N-1);
    double dt = 1e-2;
    double t_max = 1200;
    double transient_time = 200;
    int save_period = 20;
    double population_factor = 1.0; // Fattore estrazione popolazione iniziale
    time_t my_seed;
    char *name_buffer;
    FILE *fp; // Pointer to File

    name_buffer = my_char_malloc(CHAR_LENGHT); // Memoria liberata esplicitamente nel main

    kurtz_ecosystem ecosystem;
    initialise_ecosystem(&ecosystem, N, omega, lambda);

    // Create the parameter instances
    char ia_label[] = "HalfNormal";
    SymAntisymHalfNormalParams interaction_params;
    interaction_params.p_A = p_A;
    interaction_params.sigma = sigma;
    SingleParam gr_params;
    gr_params.par = 0.05;
    SingleParam cc_params;
    cc_params.par = 0.05;

    // Instantiate the generators and bind the pointers
    InteractionGenerator interaction_gen;
    interaction_gen.func = symmetric_antisymmetric_halfnormal;
    interaction_gen.params = (void *) &interaction_params; // Bind the params
    DoubleGenerator gr_generator;
    gr_generator.func = trivial_extraction;
    gr_generator.params = (void *) &gr_params;
    DoubleGenerator cc_generator;
    cc_generator.func = trivial_extraction;
    cc_generator.params = (void *) &cc_params;

    // Estraiamo N_ext volte il grafo e per ognuna di esse evolviamo N_meas volte diverse il sistema modificando ogni volta le condizioni iniziali
    for(int n=1; n<=N_ext; n++){
        seeds_from_dev_random_and_time(&my_seed); // Generate Random Seed
        srand48(my_seed); // The random seed is given
        // Extract the ecosystem
        extract_ER_ecosystem(&ecosystem, prob_link, interaction_gen, gr_generator, cc_generator);
        // MEASURES
        for(int j=(N_previous_meas+1); j<=(N_previous_meas+N_meas); j++){
            // INITIAL CONDITIONS
            for(int i=0; i<N; i++)
                ecosystem.x[i] = 1.0;
            // Run the Euler-Maruyama simulation
            int num_transient_time_steps = floor((transient_time + 0.5 * dt) / dt);
            //double t=0; // We don't actually need time at the moment
            for(int time_step = 0;  time_step < num_transient_time_steps; time_step++){
                one_step_euler_maruyama_kurtz_ecosystem(&ecosystem, dt);
                //t += dt; // We don't actually need time at the moment
            }
            // Apertura file di scrittura
            #ifdef SAVE_TO_BINARY
            snprintf(name_buffer, CHAR_LENGHT, "Trajectories_N%d_c%.2f_%s_%.2g_omega%.1f_dt%.2g.bin", N, c, ia_label, sigma, omega, dt);
            fp = my_open_writing_binary_file(name_buffer);
            #endif
            #ifndef SAVE_TO_BINARY
            snprintf(name_buffer, CHAR_LENGHT, "Trajectories_N%d_c%.2f_%s_%.2g_omega%.1f_dt%.2g.txt", N, c, ia_label, sigma, omega, dt);
            fp = my_open_writing_file(name_buffer);
            #endif
            int num_time_steps = floor((t_max - transient_time + 0.5 * dt) / dt);
            for(int time_step = 0;  time_step < num_time_steps; time_step++){
                one_step_euler_maruyama_kurtz_ecosystem(&ecosystem, dt);
                //t += dt; // We don't actually need time at the moment
                // Save
                if((time_step % save_period) == 0){
                    // Apertura file di scrittura
                    #ifdef SAVE_TO_BINARY
                    fwrite(ecosystem.x, sizeof(double), N, fp);
                    fflush(fp);
                    #endif
                    #ifndef SAVE_TO_BINARY
                    for(int i=0; i<N; i++)
                        fprintf(fp, "%lf\t", ecosystem.x[i]);
                    fprintf(fp, "\n");
                    #endif
                }
            }
            fclose(fp);
        } // Fine loop sulle misure
        //save_ecosystem(&ecosystem, "Nodes.txt", "Interactions.txt");
        reset_ecosystem(&ecosystem);
    } // Fine del ciclo sulle estrazioni
    // Pulizia memoria
    erase_ecosystem(&ecosystem); // Pulizia memoria grafo
    free(name_buffer); // Pulizia memoria stringa
    // Calcolo tempo impiegato
    clock_t end = clock(); //Clock conclusivo
    time_spent = ((float)(end - begin)) / CLOCKS_PER_SEC; //Calcolo tempo di esecuzione
    printf("\n # Il programma ha impiegato %f secondi.\n # A breve dovrebbe arrivare la mail (se opportunamente richiesto presso i nostri uffici).\n \n", time_spent);
    printf("Orario di conclusione: \n");
    system("date");

    return MY_SUCCESS;
}

/* MAIN - THE END */

/* FUNCTIONS */