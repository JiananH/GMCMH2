#include <algorithm>
#include <string>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include "util.h"


extern "C" {

  SEXP spDynLMmodgtwo(SEXP Y_r, SEXP X_r, SEXP p_r, SEXP n_r, SEXP Nt_r, SEXP coordsD_r,
         SEXP beta0Norm_r, SEXP sigmaSqIG_r, SEXP tauSqIG_r, SEXP nuUnif_r, SEXP phiUnif_r, SEXP sigmaEtaIW_r,
         SEXP betaStarting_r, SEXP phiStarting_r, SEXP sigmaSqStarting_r, SEXP tauSqStarting_r, SEXP nuStarting_r, SEXP sigmaEtaStarting_r,
         SEXP phiTuning_r, SEXP nuTuning_r, SEXP covModel_r, SEXP nSamples_r, SEXP missing_r, SEXP getFitted_r, SEXP verbose_r, SEXP nReport_r,
         SEXP radiusbeta0_r, SEXP radiusbeta_r, SEXP radiustausq_r, SEXP radiussigmasq_r, SEXP radiusphi_r, SEXP radiussigmaEta_r, SEXP prob_r){

    /*****************************************
                Common variables
    *****************************************/
    int i, j, k, l, b, s, t, info, nProtect=0;
    char const *lower = "L";
    char const *upper = "U";
    char const *nUnit = "N";
    char const *yUnit = "U";
    char const *ntran = "N";
    char const *ytran = "T";
    char const *rside = "R";
    char const *lside = "L";
    const double one = 1.0;
    const double two = 2.0;
    const double negOne = -1.0;
    const double zero = 0.0;
    const int incOne = 1;

    /*****************************************
                     Set-up
    *****************************************/
    double *Y = REAL(Y_r); //n x Nt
    double *X = REAL(X_r); //p x Nt*n
    int p = INTEGER(p_r)[0];
    int pp = p*p;
    int n = INTEGER(n_r)[0];
    int Nt = INTEGER(Nt_r)[0];
    int Ntp = Nt*p;
    int nn = n*n;
    int Ntn = Nt*n;
    double prob = REAL(prob_r)[0];


    double *coordsD = REAL(coordsD_r);

    std::string covModel = CHAR(STRING_ELT(covModel_r,0));

    //priors
    double *m0 = (double *) R_alloc(p, sizeof(double));
    F77_NAME(dcopy)(&p, REAL(VECTOR_ELT(beta0Norm_r, 0)), &incOne, m0, &incOne);

    double *sigma0 = (double *) R_alloc(pp, sizeof(double));
    F77_NAME(dcopy)(&pp, REAL(VECTOR_ELT(beta0Norm_r, 1)), &incOne, sigma0, &incOne);

    double *sigmaSqIG = REAL(sigmaSqIG_r);
    double *phiUnif = REAL(phiUnif_r);
    double *tauSqIG = REAL(tauSqIG_r);

    //matern
    double *nuUnif = NULL;
    if(covModel == "matern"){
      nuUnif = REAL(nuUnif_r);
    }

    double *SigmaEtaIW_S = (double *) R_alloc(pp, sizeof(double));
    double SigmaEtaIW_df = REAL(VECTOR_ELT(sigmaEtaIW_r, 0))[0]; SigmaEtaIW_S = REAL(VECTOR_ELT(sigmaEtaIW_r, 1));

    int nSamples = INTEGER(nSamples_r)[0];
    int *missing = INTEGER(missing_r);
    bool getFitted = static_cast<bool>(INTEGER(getFitted_r)[0]);
    int verbose = INTEGER(verbose_r)[0];
    int nReport = INTEGER(nReport_r)[0];

    int nMissing = 0;
    bool anyMissing = false;
    for(i = 0; i < Ntn; i++){
      if(missing[i] == 1){
  nMissing++;
  anyMissing = true;
      }
    }

    if(verbose){
      Rprintf("----------------------------------------\n");
      Rprintf("\tGeneral model description\n");
      Rprintf("----------------------------------------\n");
      Rprintf("Model fit with %i observations in %i time steps.\n\n", n, Nt);
      Rprintf("Number of missing observations %i.\n\n", nMissing);
      Rprintf("Number of covariates %i (including intercept if specified).\n\n", p);
      Rprintf("Using the %s spatial correlation model.\n\n", covModel.c_str());

      Rprintf("Number of MCMC samples %i.\n\n", nSamples);

      Rprintf("Priors and hyperpriors:\n");

      Rprintf("\tbeta normal:\n");
      Rprintf("\tm_0:"); printVec(m0, p);
      Rprintf("\tSigma_0:\n"); printMtrx(sigma0, p, p);
      Rprintf("\n");

      for(i = 0, j=1; i < Nt; i++, j++){
  Rprintf("\tsigma.sq_t=%i IG hyperpriors shape=%.5f and scale=%.5f\n", j, sigmaSqIG[i*2], sigmaSqIG[i*2+1]);

  Rprintf("\ttau.sq_t=%i IG hyperpriors shape=%.5f and scale=%.5f\n", j, tauSqIG[i*2], tauSqIG[i*2+1]);

  Rprintf("\tphi_t=%i Unif hyperpriors a=%.5f and b=%.5f\n", j, phiUnif[i*2], phiUnif[i*2+1]);
  if(covModel == "matern"){
    Rprintf("\tnu_t=%i Unif hyperpriors a=%.5f and b=%.5f\n", j, nuUnif[i*2], nuUnif[i*2+1]);
  }
  Rprintf("\t---\n");
      }
    }

    /*****************************************
         Set-up MCMC sample matrices etc.
    *****************************************/
    //parameters
    int nTheta, sigmaSqIndx, tauSqIndx, phiIndx, nuIndx;

    if(covModel != "matern"){
      nTheta = 3;//sigma^2, tau^2, phi
      sigmaSqIndx = 0; tauSqIndx = 1; phiIndx = 2;
    }else{
      nTheta = 4;//sigma^2, tau^2, phi, nu
      sigmaSqIndx = 0; tauSqIndx = 1; phiIndx = 2; nuIndx = 3;//sigma^2, tau^2, phi, nu
    }

    int NtnTheta = Nt*nTheta;

    double *beta0 = (double *) R_alloc(p, sizeof(double)); zeros(beta0, p);
    //printf("%d,%d,\n",beta0[0],beta0[1]);

    double *theta = (double *) R_alloc(NtnTheta, sizeof(double));
    double *beta = (double *) R_alloc(Ntp, sizeof(double));
    double *u = (double *) R_alloc(Ntn, sizeof(double)); zeros(u, Ntn);
    double *sigmaEta = (double *) R_alloc(pp, sizeof(double));

    //starting
    for(i = 0; i < Nt; i++){

      F77_NAME(dcopy)(&p, &REAL(betaStarting_r)[p*i], &incOne, &beta[p*i], &incOne); //p x Nt

      theta[nTheta*i+sigmaSqIndx] = REAL(sigmaSqStarting_r)[i]; //nTheta x Nt

      theta[nTheta*i+tauSqIndx] = REAL(tauSqStarting_r)[i];

      theta[nTheta*i+phiIndx] = logit(REAL(phiStarting_r)[i], phiUnif[i*2], phiUnif[i*2+1]);

      if(covModel == "matern"){
  theta[nTheta*i+nuIndx] = logit(REAL(nuStarting_r)[i], nuUnif[i*2], nuUnif[i*2+1]);
      }
    }

    F77_NAME(dcopy)(&pp, REAL(sigmaEtaStarting_r), &incOne, sigmaEta, &incOne);

    //tuning
    double *phiTuning = (double *) R_alloc(Nt, sizeof(double));
    double *nuTuning = NULL;
    if(covModel == "matern"){
      nuTuning = (double *) R_alloc(Nt, sizeof(double));
    }

    for(i = 0; i < Nt; i++){
      phiTuning[i] = REAL(phiTuning_r)[i];

      if(covModel == "matern"){
  nuTuning[i] = REAL(nuTuning_r)[i];
      }
    }

    //return stuff
    SEXP thetaSamples_r, beta0Samples_r, betaSamples_r, uSamples_r, sigmaEtaSamples_r, ySamples_r;
    PROTECT(thetaSamples_r = allocMatrix(REALSXP, nTheta*Nt, nSamples)); nProtect++;
    PROTECT(beta0Samples_r = allocMatrix(REALSXP, p, nSamples)); nProtect++;
    PROTECT(betaSamples_r = allocMatrix(REALSXP, p*Nt, nSamples)); nProtect++;
    PROTECT(uSamples_r = allocMatrix(REALSXP, n*Nt, nSamples)); nProtect++;
    PROTECT(sigmaEtaSamples_r = allocMatrix(REALSXP, pp, nSamples)); nProtect++;

    double *thetaSamples = REAL(thetaSamples_r);
    double *beta0Samples = REAL(beta0Samples_r);
    double *betaSamples = REAL(betaSamples_r);
    double *uSamples = REAL(uSamples_r);
    double *sigmaEtaSamples = REAL(sigmaEtaSamples_r);
    double *ySamples = NULL;

    if(getFitted){
      PROTECT(ySamples_r = allocMatrix(REALSXP, n*Nt, nSamples)); nProtect++;
      ySamples = REAL(ySamples_r);
    }

    /*****************************************
       Set-up MCMC alg. vars. matrices etc.
    *****************************************/
    int status = 1, accept = 0, batchAccept=0;
    double logMHRatio, logPostCand, logPost, logDetCand, logDet;



    double *tmp_pp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_ppp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_pppp = (double *) R_alloc(pp, sizeof(double));
    double *tmp_pp2 = (double *) R_alloc(pp, sizeof(double));
    double *tmp_p = (double *) R_alloc(p, sizeof(double));
    double *tmp_p2 = (double *) R_alloc(p, sizeof(double));
    double tmp;
    double *tmp_n = (double *) R_alloc(n, sizeof(double));
    double *tmp_n2 = (double *) R_alloc(n, sizeof(double));
    double *tmp_n3 = (double *) R_alloc(n, sizeof(double));
    double *C = (double *) R_alloc(nn, sizeof(double));
    double *C2 = (double *) R_alloc(nn, sizeof(double));
    double *C3 = (double *) R_alloc(nn, sizeof(double));
    double *gamma = (double *) R_alloc(3, sizeof(double)); //sigma^2, phi, nu

    //Sigma_0^{-1}
    F77_NAME(dpotrf)(lower, &p, sigma0, &p, &info); if(info != 0){error("c++ error: dpotrf failed 0\n");}
    F77_NAME(dpotri)(lower, &p, sigma0, &p, &info); if(info != 0){error("c++ error: dpotri failed 0.5\n");}

    //X'_tX_t
    double *XX = (double *) R_alloc(Nt*pp, sizeof(double));
    for(t = 0; t < Nt; t++){
      F77_NAME(dgemm)(ntran, ytran, &p, &p, &n, &one, &X[t*n*p], &p, &X[t*n*p], &p, &zero, &XX[t*pp], &p);
    }

    //SigmaEtaIW_S is Upsilon^{-1}
    F77_NAME(dpotrf)(lower, &p, SigmaEtaIW_S, &p, &info); if(info != 0){error("c++ error: dpotrf failed 1.1\n");}
    F77_NAME(dpotri)(lower, &p, SigmaEtaIW_S, &p, &info); if(info != 0){error("c++ error: dpotri failed 2.01\n");}

    if(verbose){
      // Rprintf("-------------------------------------------------\n");
      // Rprintf("\t\tSampling\n");
      // Rprintf("-------------------------------------------------\n");
      #ifdef Win32
        R_FlushConsole();
      #endif
    }

    GetRNGstate();

    for(s = 0; s < nSamples; s++){

      //Sigma_eta^{-1}
      F77_NAME(dpotrf)(lower, &p, sigmaEta, &p, &info); if(info != 0){error("c++ error: dpotrf1 failed\n");}
      F77_NAME(dpotri)(lower, &p, sigmaEta, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}

      //update Beta_0
      F77_NAME(dcopy)(&pp, sigmaEta, &incOne, tmp_pp, &incOne);
      F77_NAME(daxpy)(&pp, &one, sigma0, &incOne, tmp_pp, &incOne);

      //Sigma_Beta0
      F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf2 failed\n");}
      F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}

      // Rprintf("%i th sample-------------------------------------------\n",s);
      // Rprintf("\n covariance matrix tmp_pp:"); printMtrx(tmp_pp,p,p);
      F77_NAME(dcopy)(&pp, tmp_pp, &incOne, tmp_ppp, &incOne);
      // Rprintf("\n covariance matrix tmp_ppp:"); printMtrx(tmp_ppp,p,p);

      //mu_Beta0

      F77_NAME(dsymv)(lower, &p, &one, sigma0, &p, m0, &incOne, &zero, tmp_p, &incOne);
      F77_NAME(dsymv)(lower, &p, &one, sigmaEta, &p, beta, &incOne, &one, tmp_p, &incOne);//first column of beta is Beta_1
      F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &incOne, &zero, tmp_p2, &incOne);

      // Rprintf("-------------------------------------------------\n");
      // Rprintf("\tmean vector tmp_p2:"); printVec(tmp_p2,p);

      /////

     F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf3 failed\n");}


     //SECOND STRATEGY: uniform in the neighborhood
     // /************/
     //mvrnorm(&beta[t*p], tmp_p2, tmp_pp, p);
     if (s==0){
      mvrnorm(beta0, tmp_p2, tmp_pp, p);
      // double des;
      // des=dmvnorm(beta0,tmp_p2,tmp_pp,p);

     } else {

        double *radiusbeta0=REAL(radiusbeta0_r);
        double *tempbeta0 = (double *) R_alloc(p, sizeof(double));

        double *lowercurrent = (double *) R_alloc(p, sizeof(double));
        double *uppercurrent = (double *) R_alloc(p, sizeof(double));
        double *lowerpropose = (double *) R_alloc(p, sizeof(double));
        double *upperpropose = (double *) R_alloc(p, sizeof(double));

        for (int dim=0; dim<p; dim++){
              lowercurrent[dim]=beta0[dim]-radiusbeta0[dim];
              uppercurrent[dim]=beta0[dim]+radiusbeta0[dim];

            }
        int nop=2;
        double valuecurrent=0.0;
        valuecurrent=pmvnorm(nop,lowercurrent,uppercurrent,tmp_p2,tmp_pp);

        if (runif(0.0,1.0)<prob*valuecurrent){

           tempbeta0[0]=runif(lowercurrent[0],uppercurrent[0]);
           tempbeta0[1]=runif(lowercurrent[1],uppercurrent[1]);
           double despropose;
           double descurrent;
           despropose=dmvnorm(tempbeta0,tmp_p2,tmp_ppp,p);
           descurrent=dmvnorm(beta0,tmp_p2,tmp_ppp,p);

           for (int dim=0; dim<p; dim++){
              lowerpropose[dim]=tempbeta0[dim]-radiusbeta0[dim];
              upperpropose[dim]=tempbeta0[dim]+radiusbeta0[dim];

            }
           double valuepropose=0.0;
           valuepropose=pmvnorm(nop,lowerpropose,upperpropose,tmp_p2,tmp_pp);

           double accept_prob=despropose/descurrent*valuepropose/valuecurrent;
           // Rprintf("\t despropose for beta0 %.5f",despropose);
           // Rprintf("\t descurrent for beta0 %.5f",descurrent);
           // Rprintf("\t valuepropose for beta0 %.5f",valuepropose);
           // Rprintf("\t valuecurrent for beta0 %.5f",valuecurrent);
           // Rprintf("\n acceptance probability for beta0 %.5f \n",accept_prob);
           if(runif(0.0,1.0)<accept_prob){
            beta0[0]=tempbeta0[0];
            beta0[1]=tempbeta0[1];
           }
        } else {
              int *acceptmarkbeta0 = (int *) R_alloc(p, sizeof(int));
              int sumbeta0=0;
              int acceptindicator=0;
            do {
                mvrnorm(tempbeta0, tmp_p2, tmp_pp, p);
                for (int dim=0; dim<p; dim++){
                  acceptmarkbeta0[dim]=0;
                }
                for (int dim=0; dim<p; dim++){
                  if (((tempbeta0[dim]-beta0[dim])*(tempbeta0[dim]-beta0[dim]))>(radiusbeta0[dim]*radiusbeta0[dim])){
                    acceptmarkbeta0[dim]=1;
                  } else {
                    acceptmarkbeta0[dim]=0;
                  }
                }
                  sumbeta0=0;
                  for (int dim=0; dim<p; dim++){
                    sumbeta0+=acceptmarkbeta0[dim];
                  }


         if(sumbeta0>=1){
              acceptindicator=1;
            } else {
              acceptindicator=0;
            }

            }while(acceptindicator==0);

            for (int dim=0; dim<p; dim++){
              lowerpropose[dim]=tempbeta0[dim]-radiusbeta0[dim];
              upperpropose[dim]=tempbeta0[dim]+radiusbeta0[dim];

            }
           double valuepropose=0.0;
           valuepropose=pmvnorm(nop,lowerpropose,upperpropose,tmp_p2,tmp_pp);

            double accept_prob=(1-prob*valuepropose)/(1-valuepropose)*(1-valuecurrent)/(1-prob*valuecurrent);
            if (accept_prob>=1) {
              accept_prob=1;
            }

            if (runif(0.0,1.0)<accept_prob){

              beta0[0]=tempbeta0[0];
              beta0[1]=tempbeta0[1];
            }
        }
    }

       /************/

     F77_NAME(dcopy)(&p, beta0, &incOne, &beta0Samples[s*p], &incOne);

     ////////////////////
     //update y_t
     ////////////////////
     if(anyMissing || getFitted){
       for(t = 0, j = 0; t < Nt; t++){
   F77_NAME(dgemv)(ytran, &p, &n, &one, &X[t*n*p], &p, &beta[t*p], &incOne, &zero, tmp_n3, &incOne);
   for(i = 0; i < n; i++, j++){
     if(missing[j] == 1){
       Y[n*t+i] = rnorm(tmp_n3[i] + u[n*t+i], sqrt(theta[t*nTheta+tauSqIndx]));
       if(getFitted){
         ySamples[s*Ntn+t*n+i] = Y[n*t+i];
       }
     }else{
       if(getFitted){
         ySamples[s*Ntn+t*n+i] = rnorm(tmp_n3[i] + u[n*t+i], sqrt(theta[t*nTheta+tauSqIndx]));
       }
     }
   }
       }
     }
// Rprintf("Beta---------------------------------------------\n");
     for(t = 0; t < Nt; t++){

       ////////////////////
       //update B_t
       ////////////////////
       //Sigma_Beta_t
       // Rprintf("Beta%i -------------------------------------------\n",t);
       // Rprintf("\ntmp_pp:"); printMtrx(tmp_pp,p,p);
       // Rprintf("\nsigmaEta:"); printMtrx(sigmaEta,p,p);
       F77_NAME(dcopy)(&pp, sigmaEta, &incOne, tmp_pp, &incOne);
       // Rprintf("\ttmp_pp:"); printVec(tmp_pp,pp);
       if(t < (Nt-1)){
        F77_NAME(dscal)(&pp, &two, tmp_pp, &incOne);
       }
       // Rprintf("\ncopy sigmaEta tmp_pp:"); printMtrx(tmp_pp,p,p);

       tmp = 1.0/theta[t*nTheta+tauSqIndx];
       F77_NAME(daxpy)(&pp, &tmp, &XX[t*pp], &incOne, tmp_pp, &incOne);
       // Rprintf("\nadding XX tmp_pp:"); printMtrx(tmp_pp,p,p);

       F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf4 failed\n");}
       F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       // Rprintf("\tinversed tmp_pp:"); printVec(tmp_pp,pp);
       // Rprintf("\ninversed tmp_pp:"); printMtrx(tmp_pp,p,p);

       //mu_Beta_t
       for(i = 0; i < n; i++){
   tmp_n[i] = (Y[n*t+i] - u[n*t+i])*tmp;
       }
       F77_NAME(dgemv)(ntran, &p, &n, &one, &X[t*n*p], &p, tmp_n, &incOne, &zero, tmp_p, &incOne);

       if(t == 0){
   for(i = 0; i < p; i++){
     tmp_p2[i] = beta0[i]+beta[p*(t+1)+i];
   }
       }if(t == (Nt-1)){
   F77_NAME(dcopy)(&p, &beta[p*(t-1)], &incOne, tmp_p2, &incOne);
       }else{
   for(i = 0; i < p; i++){
     tmp_p2[i] = beta[p*(t-1)+i]+beta[p*(t+1)+i];
   }
       }

       F77_NAME(dsymv)(lower, &p, &one, sigmaEta, &p, tmp_p2, &incOne, &one, tmp_p, &incOne);
       F77_NAME(dsymv)(lower, &p, &one, tmp_pp, &p, tmp_p, &incOne, &zero, tmp_p2, &incOne);

      
      // Rprintf("\n covariance matrix tmp_pp:"); printMtrx(tmp_pp,p,p);
      F77_NAME(dcopy)(&pp, tmp_pp, &incOne, tmp_pppp, &incOne);
      // Rprintf("\n covariance matrix tmp_pppp:"); printMtrx(tmp_pppp,p,p);


       F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf5 failed\n");}
       // Rprintf("\ttmp_pp:"); printVec(tmp_pp,pp);

       /************/
       //mvrnorm(&beta[t*p], tmp_p2, tmp_pp, p);
      double *radiusbeta=REAL(radiusbeta_r);
      double *tempbeta = (double *) R_alloc(p, sizeof(double));
      double *currentbeta = (double *) R_alloc(p, sizeof(double));

      double *lowercurrentbeta = (double *) R_alloc(p, sizeof(double));
      double *uppercurrentbeta = (double *) R_alloc(p, sizeof(double));
      double *lowerproposebeta = (double *) R_alloc(p, sizeof(double));
      double *upperproposebeta = (double *) R_alloc(p, sizeof(double));

      for (int dim=0; dim<p; dim++){
            currentbeta[dim]=beta[t*p+dim];
            lowercurrentbeta[dim]=beta[t*p+dim]-radiusbeta[dim];
            uppercurrentbeta[dim]=beta[t*p+dim]+radiusbeta[dim];
          }
          int nop=2;
          double valuecurrentbeta=0.0;
          // Rprintf("Before---------------------------------------------\n");
          // Rprintf("\n lowercurrentbeta for beta"); printVec(lowercurrentbeta,p);
          // Rprintf("\n uppercurrentbeta for beta"); printVec(uppercurrentbeta,p);
          valuecurrentbeta=pmvnorm(nop,lowercurrentbeta,uppercurrentbeta,tmp_p2,tmp_pp);

          // Rprintf("\n lowercurrentbeta for beta"); printVec(lowercurrentbeta,p);
          // Rprintf("\n uppercurrentbeta for beta"); printVec(uppercurrentbeta,p);
          // Rprintf("After---------------------------------------------\n");

       if (runif(0.0,1.0)<prob*valuecurrentbeta){
          tempbeta[0]=runif(lowercurrentbeta[0],uppercurrentbeta[0]);
          tempbeta[1]=runif(lowercurrentbeta[1],uppercurrentbeta[1]);
          double desproposebeta;
          double descurrentbeta;
          // Rprintf("\n tmp_p2 for beta"); printVec(tmp_p2,p);
          // Rprintf("\n tmp_pp for beta"); printMtrx(tmp_pp,p,p);
          desproposebeta=dmvnorm(tempbeta,tmp_p2,tmp_pppp,p);
          descurrentbeta=dmvnorm(currentbeta,tmp_p2,tmp_pppp,p);
          // Rprintf("\n tmp_p2 for beta"); printVec(tmp_p2,p);
          // Rprintf("\n tmp_pp for beta"); printMtrx(tmp_pp,p,p);

           for (int dim=0; dim<p; dim++){

            lowerproposebeta[dim]=tempbeta[dim]-radiusbeta[dim];
            upperproposebeta[dim]=tempbeta[dim]+radiusbeta[dim];
          }

          double valueproposebeta=0.0;
          // Rprintf("\n tempbeta");printVec(tempbeta,p);
          // Rprintf("\n lowerproposebeta for beta"); printVec(lowerproposebeta,p);
          // Rprintf("\n upperproposebeta for beta"); printVec(upperproposebeta,p);
          // // Rprintf("\n beta");printVec(beta,p);
          // Rprintf("\n lowercurrentbeta for beta"); printVec(lowercurrentbeta,p);
          // Rprintf("\n uppercurrentbeta for beta"); printVec(uppercurrentbeta,p);
          valueproposebeta=pmvnorm(nop,lowerproposebeta,upperproposebeta,tmp_p2,tmp_pp);


           // Rprintf("\n despropose for beta %.5f",desproposebeta);
           // Rprintf("\n descurrent for beta %.5f",descurrentbeta);
           // Rprintf("\n valuepropose for beta %.5f",valueproposebeta);
           // Rprintf("\n valuecurrent for beta %.5f",valuecurrentbeta);
           double accept_prob_beta1=desproposebeta/descurrentbeta*valueproposebeta/valuecurrentbeta;
           // Rprintf("\n acceptance probability for beta %.5f \n",accept_prob_beta1);
            if (runif(0.0,1.0)<accept_prob_beta1){
           beta[t*p]=tempbeta[0];
           beta[t*p+1]=tempbeta[1];
          }
       } else {

          int *acceptmarkbeta = (int *) R_alloc(p, sizeof(int));
          int sumbeta=0;
          int acceptindicator=0;
          do {
            mvrnorm(tempbeta, tmp_p2, tmp_pp, p);
             for (int dim=0; dim<p; dim++){
               acceptmarkbeta[dim]=0;
             }
              for (int dim=0; dim<p; dim++){
                if (((tempbeta[dim]-beta[t*p+dim])*(tempbeta[dim]-beta[t*p+dim]))>(radiusbeta[dim]*radiusbeta[dim])){
                  acceptmarkbeta[dim]=1;
                } else {
                  acceptmarkbeta[dim]=0;
                }
              }
                sumbeta=0;
                for (int dim=0; dim<p; dim++){
                  sumbeta=sumbeta+acceptmarkbeta[dim];
                }
                if(sumbeta>0){
                  acceptindicator=1;
                } else {
                  acceptindicator=0;
                }

           }while(acceptindicator==0);

            for (int dim=0; dim<p; dim++){
              lowerproposebeta[dim]=tempbeta[dim]-radiusbeta[dim];
              upperproposebeta[dim]=tempbeta[dim]+radiusbeta[dim];
            }
              double valueproposebeta2=0.0;
              valueproposebeta2=pmvnorm(nop,lowerproposebeta,upperproposebeta,tmp_p2,tmp_pp);
              double accept_prob_beta2=(1-prob*valueproposebeta2)/(1-valueproposebeta2)*(1-valuecurrentbeta)/(1-prob*valuecurrentbeta);
            if (runif(0.0,1.0)<accept_prob_beta2){

              beta[t*p]=tempbeta[0];
              beta[t*p+1]=tempbeta[1];
            }
       }






       /************/

       ////////////////////
       //update u_t
       ////////////////////
       gamma[0] = theta[t*nTheta+sigmaSqIndx];
       gamma[1] = logitInv(theta[t*nTheta+phiIndx], phiUnif[t*2], phiUnif[t*2+1]);
       if(covModel == "matern"){
         gamma[2] = logitInv(theta[t*nTheta+nuIndx], nuUnif[t*2], nuUnif[t*2+1]);
       }

       spCovLT(coordsD, n, gamma, covModel, C);
       //printf("%d,%d,\n",C[0],C[1]);

       F77_NAME(dpotrf)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotrf6 failed\n");}
       F77_NAME(dpotri)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       F77_NAME(dgemv)(ytran, &p, &n, &one, &X[t*n*p], &p, &beta[t*p], &incOne, &zero, tmp_n3, &incOne);

       if(t < (Nt-1)){

         //t+1
         gamma[0] = theta[(t+1)*nTheta+sigmaSqIndx];
         gamma[1] = logitInv(theta[(t+1)*nTheta+phiIndx], phiUnif[(t+1)*2], phiUnif[(t+1)*2+1]);
         if(covModel == "matern"){
           gamma[2] = logitInv(theta[(t+1)*nTheta+nuIndx], nuUnif[(t+1)*2], nuUnif[(t+1)*2+1]);
         }

         spCovLT(coordsD, n, gamma, covModel, C2);

         F77_NAME(dpotrf)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotrf7 failed\n");}
         F77_NAME(dpotri)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

         //mu
         F77_NAME(dsymv)(lower, &n, &one, C2, &n, &u[(t+1)*n], &incOne, &zero, tmp_n, &incOne);

         if(t > 0){
           F77_NAME(dsymv)(lower, &n, &one, C, &n, &u[(t-1)*n], &incOne, &one, tmp_n, &incOne);
         }

         for(i = 0; i < n; i++){
           tmp_n[i] += (Y[t*n+i] - tmp_n3[i])*tmp;
         }

         //sigma
         F77_NAME(daxpy)(&nn, &one, C, &incOne, C2, &incOne);

         for(i = 0; i < n; i++){
           C2[i*n+i] += tmp;
         }

         F77_NAME(dpotrf)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotrf8 failed\n");}
         F77_NAME(dpotri)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

         //finish mu
         F77_NAME(dsymv)(lower, &n, &one, C2, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);

         F77_NAME(dpotrf)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotrf9 failed\n");}
         mvrnorm(&u[t*n], tmp_n2, C2, n);

       }else{//t==Nt

         F77_NAME(dcopy)(&nn, C, &incOne, C2, &incOne);

         //mu
         F77_NAME(dsymv)(lower, &n, &one, C2, &n, &u[(t-1)*n], &incOne, &zero, tmp_n, &incOne);

         for(i = 0; i < n; i++){
           tmp_n[i] += (Y[t*n+i] - tmp_n3[i])*tmp;
         }

         //sigma
         for(i = 0; i < n; i++){
           C2[i*n+i] += tmp;
         }

         F77_NAME(dpotrf)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotrf10 failed\n");}
         F77_NAME(dpotri)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

         //finish mu
         F77_NAME(dsymv)(lower, &n, &one, C2, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);

         F77_NAME(dpotrf)(lower, &n, C2, &n, &info); if(info != 0){error("c++ error: dpotrf11 failed\n");}


         /************/
        mvrnorm(&u[t*n], tmp_n2, C2, n);
       //mvrnorm(&beta[t*p], tmp_p2, tmp_pp, p);
       // double radius=0.0002;
       // double *temp = (double *) R_alloc(n, sizeof(double));
       // int *acceptmark = (int *) R_alloc(n, sizeof(int));
       // int sum=0;

       // do {
       //  mvrnorm(temp, tmp_n2, C2, n);
       //   for (int dim=0; dim<n; dim++){
       //     acceptmark[dim]=0;
       //   }
       //    for (int dim=0; dim<n; dim++){
       //      if (((temp[dim]-beta[t*p+dim])*(temp[dim]-beta[t*p+dim]))>(radius*radius)){
       //        acceptmark[dim]=1;
       //      } else {
       //        acceptmark[dim]=0;
       //      }
       //      sum=0;
       //      for (int dim=0; dim<n; dim++){
       //        sum=sum+acceptmark[dim];
       //      }
       //      //printf("%d,%d,%d\n",acceptmark[0],acceptmark[1],sum);
       //   }
       // }while(sum==0);

       // for (int i=0; i<n; i++){
       //  u[t*n+i]=temp[i];
       // }
       // /************/
       }

       ////////////////////
       //update tau^2
       ////////////////////
       for(i = 0; i < n; i++){
         tmp_n[i] = Y[t*n+i] - tmp_n3[i] - u[t*n+i];
       }
       // theta[t*nTheta+tauSqIndx] = 1.0/rgamma(tauSqIG[t*2]+n/2.0,
       //                1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)));

       /************/

       double radiustausq=REAL(radiustausq_r)[0];
       double temptausq;
       int acceptmarktausq=0;
       double valuecurrenttausq=pgamma(1.0/(theta[t*nTheta+tauSqIndx]+radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0)-pgamma(1.0/(theta[t*nTheta+tauSqIndx]-radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0);


       if (runif(0.0,1.0)<prob*valuecurrenttausq){

        double lowercurrenttausq=theta[t*nTheta+tauSqIndx]-radiustausq;
        double uppercurrenttausq=theta[t*nTheta+tauSqIndx]+radiustausq;

        temptausq=runif(lowercurrenttausq,uppercurrenttausq);
        double valueproposetausq1=pgamma(1.0/(temptausq+radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0)-pgamma(1.0/(temptausq-radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0);

        double accept_prob_tausq1=dgamma(1.0/temptausq,tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),0)/dgamma(1.0/theta[t*nTheta+tauSqIndx],tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),0)*valueproposetausq1/valuecurrenttausq;

        if (runif(0.0,1.0)<accept_prob_tausq1){
          theta[t*nTheta+tauSqIndx] = temptausq;
        }

       } else {
         do {

         temptausq=1.0/rgamma(tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)));

            if (((temptausq-theta[t*nTheta+tauSqIndx])*(temptausq-theta[t*nTheta+tauSqIndx]))>(radiustausq*radiustausq)){
              acceptmarktausq=1;
            } else{
              acceptmarktausq=0;
            }

       }  while(acceptmarktausq==0);

       double valueproposetausq2=pgamma(1.0/(temptausq+radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0)-pgamma(1.0/(temptausq-radiustausq),tauSqIG[t*2]+n/2.0,1.0/(tauSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n, &incOne)),1,0);

       double accept_prob_tausq2=(1.0-prob*valueproposetausq2)/(1.0-valueproposetausq2)*(1.0-valuecurrenttausq)/(1.0-prob*valuecurrenttausq);
            if (accept_prob_tausq2>=1.0) {
              accept_prob_tausq2=1.0;
            }
            if (runif(0.0,1.0)<accept_prob_tausq2){
              theta[t*nTheta+tauSqIndx] = temptausq;
            }
       }

       /************/


       ////////////////////
       //update sigma^2
       ////////////////////
       F77_NAME(dcopy)(&n, &u[n*t], &incOne, tmp_n, &incOne);
       if(t > 0){
         for(i = 0; i < n; i++){
           tmp_n[i] = tmp_n[i] - u[n*(t-1)+i];
         }
       }

       F77_NAME(dsymv)(lower, &n, &one, C, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);

       // theta[t*nTheta+sigmaSqIndx] = 1.0/rgamma(sigmaSqIG[t*2]+n/2.0,
       //            1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]));


       // /************/
       double radiussigmasq=REAL(radiussigmasq_r)[0];
       double tempsigmasq;
       int acceptmarksigmasq=0;

       double valuecurrentsigmasq=pgamma(1.0/(theta[t*nTheta+tauSqIndx]+radiustausq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0)-pgamma(1.0/(theta[t*nTheta+tauSqIndx]-radiustausq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0);


       if (runif(0.0,1.0)<prob*valuecurrentsigmasq){

        double lowercurrentsigmasq=theta[t*nTheta+sigmaSqIndx]-radiussigmasq;
        double uppercurrentsigmasq=theta[t*nTheta+sigmaSqIndx]+radiussigmasq;

        tempsigmasq=runif(lowercurrentsigmasq,uppercurrentsigmasq);

        double valueproposesigmasq1=pgamma(1.0/(tempsigmasq+radiussigmasq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0)-pgamma(1.0/(temptausq-radiustausq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0);

        double accept_prob_sigmasq1=dgamma(1.0/tempsigmasq,sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),0)/dgamma(1.0/theta[t*nTheta+sigmaSqIndx],sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),0)*valueproposesigmasq1/valuecurrentsigmasq;

        if (runif(0.0,1.0)<accept_prob_sigmasq1){
          theta[t*nTheta+sigmaSqIndx] = tempsigmasq;
        }
       } else {
         do {
         tempsigmasq=1.0/rgamma(sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]));
            if (((tempsigmasq-theta[t*nTheta+sigmaSqIndx])*(tempsigmasq-theta[t*nTheta+sigmaSqIndx]))>(radiussigmasq*radiussigmasq)){
              acceptmarksigmasq=1;
            } else{
                acceptmarksigmasq=0;
              }

       }  while(acceptmarksigmasq==0);
        double valueproposesigmasq2=pgamma(1.0/(tempsigmasq+radiussigmasq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0)-pgamma(1.0/(temptausq-radiustausq),sigmaSqIG[t*2]+n/2.0, 1.0/(sigmaSqIG[t*2+1]+0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne)*theta[t*nTheta+sigmaSqIndx]),1,0);

        double accept_prob_sigmasq2=(1.0-prob*valueproposesigmasq2)/(1.0-valueproposesigmasq2)*(1.0-valuecurrentsigmasq)/(1.0-prob*valuecurrentsigmasq);
            if (accept_prob_sigmasq2>=1.0) {
              accept_prob_sigmasq2=1.0;
            }
            if (runif(0.0,1.0)<accept_prob_sigmasq2){
              theta[t*nTheta+sigmaSqIndx] = tempsigmasq;
            }
       }



       //theta[t*nTheta+sigmaSqIndx] = tempsigmasq;

       /************/

       ////////////////////
       //update phi
       ////////////////////
       if(t > 0){
          for(i = 0; i < n; i++){
            tmp_n[i] = u[n*t+i]-u[n*(t-1)+i];
          }
       }else{
          F77_NAME(dcopy)(&n, &u[n*t], &incOne, tmp_n, &incOne);
       }

       //current
       gamma[0] = theta[t*nTheta+sigmaSqIndx];
       gamma[1] = logitInv(theta[t*nTheta+phiIndx], phiUnif[t*2], phiUnif[t*2+1]);
       if(covModel == "matern"){
          gamma[2] = logitInv(theta[t*nTheta+nuIndx], nuUnif[t*2], nuUnif[t*2+1]);
       }

       spCovLT(coordsD, n, gamma, covModel, C);

       logDet = 0;
       F77_NAME(dpotrf)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotrf12 failed\n");}
       for(i = 0; i < n; i++) logDet += 2*log(C[i*n+i]);
       F77_NAME(dpotri)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       F77_NAME(dsymv)(lower, &n, &one, C, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);
       logPost = -0.5*logDet-0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne);

       logPost += log(gamma[1] - phiUnif[t*2]) + log(phiUnif[t*2+1] - gamma[1]);
       if(covModel == "matern"){
          logPost += log(gamma[2] - nuUnif[t*2]) + log(nuUnif[t*2+1] - gamma[2]);
       }

       //cand
       gamma[0] = theta[t*nTheta+sigmaSqIndx];
       gamma[1] = logitInv(rnorm(theta[t*nTheta+phiIndx], phiTuning[t]), phiUnif[t*2], phiUnif[t*2+1]);
       if(covModel == "matern"){
          gamma[2] = logitInv(rnorm(theta[t*nTheta+nuIndx], nuTuning[t]), nuUnif[t*2], nuUnif[t*2+1]);
       }

       spCovLT(coordsD, n, gamma, covModel, C);

       logDetCand = 0;
       F77_NAME(dpotrf)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotrf13 failed\n");}
       for(i = 0; i < n; i++) logDetCand += 2*log(C[i*n+i]);
       F77_NAME(dpotri)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       F77_NAME(dsymv)(lower, &n, &one, C, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);
       logPostCand = -0.5*logDetCand-0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne);

       logPostCand += log(gamma[1] - phiUnif[t*2]) + log(phiUnif[t*2+1] - gamma[1]);
       if(covModel == "matern"){
          logPostCand += log(gamma[2] - nuUnif[t*2]) + log(nuUnif[t*2+1] - gamma[2]);
       }

       logMHRatio = logPostCand - logPost;

       if(runif(0.0,1.0) <= exp(logMHRatio)){

          theta[t*nTheta+phiIndx] = logit(gamma[1], phiUnif[t*2], phiUnif[t*2+1]);
          if(covModel == "matern"){
            theta[t*nTheta+nuIndx] = logit(gamma[2], nuUnif[t*2], nuUnif[t*2+1]);
          }
          accept++;
          batchAccept++;
       }

       /************/

       // double radiusphi=REAL(radiusphi_r)[0];
       // double tempphi;
       // int acceptmarkphi=0;

       // //printf("%f,\n",radiusphi);

       // if(t > 0){
       //   for(i = 0; i < n; i++){
       //     tmp_n[i] = u[n*t+i]-u[n*(t-1)+i];
       //   }
       // }else{
       //   F77_NAME(dcopy)(&n, &u[n*t], &incOne, tmp_n, &incOne);
       // }

       // //current
       // gamma[0] = theta[t*nTheta+sigmaSqIndx];
       // gamma[1] = logitInv(theta[t*nTheta+phiIndx], phiUnif[t*2], phiUnif[t*2+1]);
       // if(covModel == "matern"){
       //   gamma[2] = logitInv(theta[t*nTheta+nuIndx], nuUnif[t*2], nuUnif[t*2+1]);
       // }

       // spCovLT(coordsD, n, gamma, covModel, C);

       // logDet = 0;
       // F77_NAME(dpotrf)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotrf12 failed\n");}
       // for(i = 0; i < n; i++) logDet += 2*log(C[i*n+i]);
       // F77_NAME(dpotri)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       // F77_NAME(dsymv)(lower, &n, &one, C, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);
       // logPost = -0.5*logDet-0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne);

       // logPost += log(gamma[1] - phiUnif[t*2]) + log(phiUnif[t*2+1] - gamma[1]);
       // if(covModel == "matern"){
       //   logPost += log(gamma[2] - nuUnif[t*2]) + log(nuUnif[t*2+1] - gamma[2]);
       // }

       // //cand
       // do{
       //   gamma[0] = theta[t*nTheta+sigmaSqIndx];
       //   gamma[1] = logitInv(rnorm(theta[t*nTheta+phiIndx], phiTuning[t]), phiUnif[t*2], phiUnif[t*2+1]);

       //   spCovLT(coordsD, n, gamma, covModel, C);

       //   logDetCand = 0;
       //   F77_NAME(dpotrf)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotrf13 failed\n");}
       //   for(i = 0; i < n; i++) logDetCand += 2*log(C[i*n+i]);
       //   F77_NAME(dpotri)(lower, &n, C, &n, &info); if(info != 0){error("c++ error: dpotri failed\n");}

       //   F77_NAME(dsymv)(lower, &n, &one, C, &n, tmp_n, &incOne, &zero, tmp_n2, &incOne);
       //   logPostCand = -0.5*logDetCand-0.5*F77_NAME(ddot)(&n, tmp_n, &incOne, tmp_n2, &incOne);

       //   logPostCand += log(gamma[1] - phiUnif[t*2]) + log(phiUnif[t*2+1] - gamma[1]);

       //   logMHRatio = logPostCand - logPost;

       //   if(runif(0.0,1.0) <= exp(logMHRatio)){

       //     tempphi = logit(gamma[1], phiUnif[t*2], phiUnif[t*2+1]);
       //     accept++;
       //     batchAccept++;
       //    }
       //  acceptmarkphi=0;
       // if(((tempphi-theta[t*nTheta+phiIndx])*(tempphi-theta[t*nTheta+phiIndx]))>(radiusphi*radiusphi)){
       //        acceptmarkphi=1;

       //      } else {
       //        if (runif(0.0,1.0)<prob){
       //          acceptmarkphi=1;
       //        } else{
       //          acceptmarkphi=0;
       //        }
       //      }
       // }  while(acceptmarkphi==0);

       // theta[t*nTheta+phiIndx] = tempphi;




       // if(runif(0.0,1.0) <= exp(logMHRatio)){

       // do {
       //   tempphi = logit(gamma[1], phiUnif[t*2], phiUnif[t*2+1]);

       //  if (((tempphi-theta[t*nTheta+phiIndx])*(tempphi-theta[t*nTheta+phiIndx]))>(radiussigmasq*radiussigmasq)){
       //        acceptmarkphi=1;
       //      } else {
       //        acceptmarkphi=0;
       //      }
       // }  while(acceptmarkphi==0);

       //    theta[t*nTheta+phiIndx] = 1.0/tempphi;
       //    accept++;
       //   batchAccept++;
       //  }
       /************/

       R_CheckUserInterrupt();
     }//end Nt

     F77_NAME(dcopy)(&Ntp, beta, &incOne, &betaSamples[s*Ntp], &incOne);
     F77_NAME(dcopy)(&Ntn, u, &incOne, &uSamples[s*Ntn], &incOne);
     F77_NAME(dcopy)(&NtnTheta, theta, &incOne, &thetaSamples[s*NtnTheta], &incOne);

     //update Sigma_eta
     for(i = 0; i < p; i++){
       tmp_p[i] = beta[i]-beta0[i];
     }
     F77_NAME(dgemm)(ntran, ytran, &p, &p, &incOne, &one, tmp_p, &p, tmp_p, &p, &zero, tmp_pp, &p);

     for(t = 1; t < Nt; t++){
       for(i = 0; i < p; i++){
       tmp_p[i] = beta[p*t+i]-beta[p*(t-1)+i];
       }
       F77_NAME(dgemm)(ntran, ytran, &p, &p, &incOne, &one, tmp_p, &p, tmp_p, &p, &one, tmp_pp, &p);
     }

     F77_NAME(daxpy)(&pp, &one, SigmaEtaIW_S, &incOne, tmp_pp, &incOne);


     rwish(tmp_pp, SigmaEtaIW_df+Nt, p, sigmaEta, tmp_pp2, 1);
     /********************/
     // double *tempsigmaEta = (double *) R_alloc(pp, sizeof(double));
     // //double radiussigmaEta=171.67;
     // //double radiussigmaEta=442.66;
     // //double radiussigmaEta=744.82*2;
     // double radiussigmaEta=REAL(radiussigmaEta_r)[0];
     // int acceptmarksigmaEta=0;

     // //printf("%f,\n",radiussigmaEta);

     // do {
     //  rwish(tmp_pp, SigmaEtaIW_df+Nt, p, tempsigmaEta, tmp_pp2, 1);
     //  double sum=0;

     //  for (int dim=0; dim<pp; dim++){
     //    sum=sum+(tempsigmaEta[dim]-sigmaEta[dim])*(tempsigmaEta[dim]-sigmaEta[dim]);
     //  }

     //  acceptmarksigmaEta=0;
     //  if (sum>(radiussigmaEta)){
     //          acceptmarksigmaEta=1;
     //        } else {
     //          if (runif(0.0,1.0)<prob){
     //            acceptmarksigmaEta=1;
     //          } else{
     //            acceptmarksigmaEta=0;
     //          }
     //        }
     //    //printf("%d,%d\n",acceptmarksigmaEta,sum);
     // } while(acceptmarksigmaEta==0);

     // sigmaEta=tempsigmaEta;

      /********************/

     // F77_NAME(dpotrf)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
     // F77_NAME(dpotri)(lower, &p, tmp_pp, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}

     // for(i = 1; i < p; i++){
     //   for(j = 0; j < i; j++){
     //    tmp_pp[i*p+j] = tmp_pp[j*p+i];
     //   }
     // }
     // riwishart(tmp_pp, SigmaEtaIW_df+Nt, p, C, C2, C3, sigmaEta);

     F77_NAME(dcopy)(&pp, sigmaEta, &incOne, &sigmaEtaSamples[s*pp], &incOne);


     //report
     if(verbose){
       if(status == nReport){
   Rprintf("Sampled: %i of %i, %3.2f%%\n", s, nSamples, 100.0*s/nSamples);
   Rprintf("Report interval Mean Metrop. Acceptance rate: %3.2f%%\n", 100.0*batchAccept/(nReport*Nt));
   Rprintf("Overall Metrop. Acceptance rate: %3.2f%%\n", 100.0*accept/(s*Nt));
   Rprintf("-------------------------------------------------\n");
         #ifdef Win32
   R_FlushConsole();
         #endif
   status = 0;
   batchAccept = 0;
       }
     }
     status++;

    }//end sample loop

    PutRNGstate();

    //untransform variance variables
    for(s = 0; s < nSamples; s++){
      for(t = 0; t < Nt; t++){
      thetaSamples[s*NtnTheta+t*nTheta+phiIndx] = logitInv(thetaSamples[s*NtnTheta+t*nTheta+phiIndx], phiUnif[t*2], phiUnif[t*2+1]);

      if(covModel == "matern"){
    thetaSamples[s*NtnTheta+t*nTheta+nuIndx] = logitInv(thetaSamples[s*NtnTheta+t*nTheta+nuIndx], nuUnif[t*2], nuUnif[t*2+1]);
      }
      }
    }

    // //make return object
    SEXP result_r, resultName_r;
    int nResultListObjs = 5;
    if(getFitted){
      nResultListObjs++;
    }

    PROTECT(result_r = allocVector(VECSXP, nResultListObjs)); nProtect++;
    PROTECT(resultName_r = allocVector(VECSXP, nResultListObjs)); nProtect++;

    //samples
    SET_VECTOR_ELT(result_r, 0, beta0Samples_r);
    SET_VECTOR_ELT(resultName_r, 0, mkChar("p.beta.0.samples"));

    SET_VECTOR_ELT(result_r, 1, betaSamples_r);
    SET_VECTOR_ELT(resultName_r, 1, mkChar("p.beta.samples"));

    SET_VECTOR_ELT(result_r, 2, sigmaEtaSamples_r);
    SET_VECTOR_ELT(resultName_r, 2, mkChar("p.sigma.eta.samples"));

    SET_VECTOR_ELT(result_r, 3, uSamples_r);
    SET_VECTOR_ELT(resultName_r, 3, mkChar("p.u.samples"));

    SET_VECTOR_ELT(result_r, 4, thetaSamples_r);
    SET_VECTOR_ELT(resultName_r, 4, mkChar("p.theta.samples"));

    if(getFitted){
      SET_VECTOR_ELT(result_r, 5, ySamples_r);
      SET_VECTOR_ELT(resultName_r, 5, mkChar("p.y.samples"));
    }

    namesgets(result_r, resultName_r);

    //unprotect
    UNPROTECT(nProtect);

    return(result_r);
  }
}
