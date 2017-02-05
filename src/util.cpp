#include <string>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>
#include "util.h"

double pmvnorm(int dim, double *lowerinput, double *upperinput, double *mean, double *cholCov){

  double alpha = 2.5;
  double eps=0.001;
  //int Nmax=50+1;

  // Rprintf("\tPMVNORM LOWER BEFORE:"); printVec(lower,dim);
  // Rprintf("\tPMVNORM UPPER BEFORE:"); printVec(upper,dim);
  double *lower = (double *) R_alloc(dim, sizeof(double));
  double *upper = (double *) R_alloc(dim, sizeof(double));

  for (int i=0;i<dim;i++){
    lower[i]=lowerinput[i]-mean[i];
    upper[i]=upperinput[i]-mean[i];
  }
  // // compute the scaled lower and upper bound
  // F77_NAME(daxpy)(&dim, &negone, mean, &inc, lower, &inc);
  // F77_NAME(daxpy)(&dim, &negone, mean, &inc, upper, &inc);
  // Rprintf("\tPMVNORM LOWER AFTER:"); printVec(lower,dim);
  // Rprintf("\tPMVNORM UPPER AFTER:"); printVec(upper,dim);

  double *y = (double *) R_alloc(dim, sizeof(double));
  double *d = (double *) R_alloc(dim, sizeof(double));
  double *e = (double *) R_alloc(dim, sizeof(double));
  double *f = (double *) R_alloc(dim, sizeof(double));
  double *w = (double *) R_alloc((dim-1), sizeof(double));
  double *ansc = (double *) R_alloc(dim, sizeof(double));

  double Varsum = 0.0;
  double Intsum = 0.0;
  d[0]=pnorm(lower[0]/cholCov[0],0.0,1.0,1,0);
  e[0]=pnorm(upper[0]/cholCov[0],0.0,1.0,1,0);
  f[0]=e[0]-d[0];

  // Rprintf("\t y:"); printVec(y,dim);
  // Rprintf("\t d:"); printVec(d,dim);
  // Rprintf("\t e:"); printVec(e,dim);
  // Rprintf("\t f:"); printVec(f,dim);
  // Rprintf("\t w:"); printVec(w,(dim-1));
  // Rprintf("\t ansc:"); printVec(ansc,dim);

  //for (int i = 0; i < Nmax; i++){
  int N=1;
  double error=10.0;
  double product = 0.0;
  //for (int N=1; N<Nmax; N++){
  do{
    for (int j = 0; j < (dim-1); j++){
      w[j]=runif(0.0, 1.0);
    }
    for (int i = 1; i < dim; i++){
      y[i-1]=qnorm(d[i-1]+w[i-1]*(e[i-1]-d[i-1]),0.0,1.0,1,0);
      //ansc<-ith row of cholCov from the first element to the (i-1)th
      for (int l = 0; l < dim; l++){
        ansc[l]=cholCov[l*dim+1];
      }
      for (int q = i; q < dim; q++){
        ansc[q]=0;
      }
      //product of ansc and y
      product = 0.0;
      for (int m = 0; m < dim; m++){
        product = product + ansc[m]*y[m];
      }
      //Rprintf("\t error %0.5f", error);
      d[i]=pnorm((lower[i]-product)/cholCov[i*dim+i],0.0,1.0,1,0);
      e[i]=pnorm((upper[i]-product)/cholCov[i*dim+i],0.0,1.0,1,0);
      f[i]=(e[i]-d[i])*(f[i-1]);
    }
  // Rprintf("\t y:"); printVec(y,dim);
  // Rprintf("\t d:"); printVec(d,dim);
  // Rprintf("\t e:"); printVec(e,dim);
  // Rprintf("\t f:"); printVec(f,dim);
  // Rprintf("\t w:"); printVec(w,(dim-1));
  // Rprintf("\t ansc:"); printVec(ansc,dim);
  // Rprintf("\t product %0.5f \n", product);

    double delta=(f[dim-1]-Intsum)/N;
    // Rprintf("\t PROB %0.5f", Intsum);
    // Rprintf("\t delta %0.5f", delta);
    Intsum += delta;
    Varsum = (N-2)*Varsum/N + delta*delta;
    error = alpha*sqrt(Varsum);
    // Rprintf("\t error %0.5f", error);
    // Rprintf("\t N %d", N);

    // Rprintf("\t PROB %0.5f", Intsum);
    // Rprintf("\t error %0.5f \n", error);
  N = N + 1;
  }while(error>eps);
  //}
  return Intsum;

}

// double dmvnorm(double *xval, double *mu, double *cholCov, int dim){
//   double *xvalmu = (double *) R_alloc(dim, sizeof(double));
//   int inc = 1;
//   double one = 1.0;
//   double zero = 0.0;
//   double des;
//   for(int i = 0; i < dim; i++){
//     xvalmu[i] = xval[i] - mu[i];
//   }
//   //printVec(xvalmu,dim);
//   F77_NAME(dtrmv)("L", "N", "N", &dim, cholCov, &dim, xvalmu, &inc);
//   //printVec(xvalmu,dim);
//   F77_NAME(dgemm)("T", "N", &dim, &dim, &inc, &one, xvalmu, &dim, xvalmu, &dim, &zero, &des, &dim);
//   return des;
// }
double dmvnorm(double *xval, double *mu, double *cholCov, int dim){
  double *xvalmu = (double *) R_alloc(dim, sizeof(double));
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  double *des = (double *) R_alloc(dim, sizeof(double));
  for(int i = 0; i < dim; i++){
    xvalmu[i] = xval[i] - mu[i];
  }
  // Rprintf("\t xval"); printVec(xval,dim);
  // Rprintf("\t mu"); printVec(mu,dim);
  // Rprintf("\t xvalmu"); printVec(xvalmu,dim);
  // Rprintf("\t cholCov"); printMtrx(cholCov,dim,dim);
  F77_NAME(dtrmv)("L", "N", "N", &dim, cholCov, &dim, xvalmu, &inc);
  // Rprintf("\t cholCov*xvalmu"); printVec(xvalmu,dim);
  F77_NAME(dgemm)("T", "N", &dim, &dim, &inc, &one, xvalmu, &dim, xvalmu, &dim, &zero, des, &dim);
  // Rprintf("\t des"); printVec(des,dim);

  double desval=exp(-0.5*des[0]);
  return desval;
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim){

  int i;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;

  //make some std norm draws
  for(i = 0; i < dim; i++)
    des[i] = rnorm(0.0, 1.0);

  //mult this vector by the lower triangle of the cholCov
  F77_NAME(dtrmv)("L", "N", "N", &dim, cholCov, &dim, des, &inc);

  //add the mean to the result
  F77_NAME(daxpy)(&dim, &one, mu, &inc, des, &inc);

  //check if the proposed value is within Bi
  //return des;
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim, bool upper){

  int i;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;

  //make some std norm draws
  for(i = 0; i < dim; i++)
    des[i] = rnorm(0.0, 1.0);

  //mult this vector by the lower triangle of the cholCov
  if(upper)
    F77_NAME(dtrmv)("U", "T", "N", &dim, cholCov, &dim, des, &inc);
  else
    F77_NAME(dtrmv)("L", "N", "N", &dim, cholCov, &dim, des, &inc);

  //add the mean to the result
  F77_NAME(daxpy)(&dim, &one, mu, &inc, des, &inc);

  //return des;

}

SEXP getList(SEXP list, const char *str){
  SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);
  int i;

  for (i = 0; i < length(list); i++)
    if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
      elmt = VECTOR_ELT(list, i);
      break;
    }

  if(elmt == R_NilValue){
    Rprintf("\nlist element %s not found\n", str);
  }

  return elmt;
}

SEXP getGetList(SEXP list, const char *str1, const char *str2){
  SEXP list2 = getList(list, str1);
  return getList(list2, str2);
}


void zeros(double *x, int length){
  for(int i = 0; i < length; i++)
    x[i] = 0.0;
}

void zeros(int *x, int length){
  for(int i = 0; i < length; i++)
    x[i] = 0;
}

void iden(double *x, int &nrow){

  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < nrow; j++){
      if(i != j)
	x[j*nrow+i] = 0.0;
      else
	x[j*nrow+i] = 1.0;
    }
  }

}

void kron(double *a, int &dima1, int &dima2,
	  double *b, int &dimb1, int &dimb2,
	  double *c, int &dimc1, int &dimc2){

  int i, j, k, l;

  for (k = 0; k < dima1; k++) {
    for (l = 0; l < dima2; l++) {
      for (i = 0; i < dimb1; i++) {
	for (j = 0; j < dimb2; j++) {
	  c[(l*dimb2+j)*dimc1+(k*dimb1+i)] = a[l*dima1+k] * b[j*dimb1+i];
	}
      }
    }
  }
}

void setLowerChol(double *A, double *S, int dim){
  int i, j, k;

  zeros(A, dim*dim);
  for(i = 0, k = 0; i < dim; i++){
    for(j = i; j < dim; j++, k++){
      A[i*dim+j] = S[k];
    }
  }
}

double dTNorm(double x, double mu, double sd, double a, double b){
  if(x < a || x > b)
    return 0.0;
  else
    return dnorm(x, mu, sd, false)/(pnorm(b, mu, sd, true, false) - pnorm(a, mu, sd, true, false));
}

void diagmm(int &nrow_b, int &ncol_b, double *a, double *b, double *c){
  for(int i = 0; i < nrow_b; i++){
    for(int j = 0; j < ncol_b; j++){
      c[j*nrow_b+i] = b[j*nrow_b+i] * a[i];
    }
  }
}

void subsetCovRow(double *x, int n, int p, int begin, int end, double *cov, double *means){

  int nSubset = end-begin+1;
  int i,j,k;

  for(i = 0; i < p; i++){
    means[i] = 0.0;
    for(j = 0; j < p; j++){
      cov[j*p+i] = 0.0;
    }
  }


  for(i = 0; i < p; i++){
    for(j = 0; j < nSubset; j++){
      means[i] += x[(i*n)+(begin+j)];
    }
    means[i] = means[i]/nSubset;
  }


  for(i = 0; i < p; i++){
    for(j = i; j < p; j++){
      for(k = 0; k < nSubset; k++){
	cov[i*p+j] += (x[(i*n)+(begin+k)]-means[i])*(x[(j*n)+(begin+k)]-means[j]);
      }
      cov[i*p+j] = cov[i*p+j]/(nSubset-1);
    }
  }

}


void subsetCovCol(double *x, int p, int begin, int end, double *cov, double *means){

  int nSubset = end-begin+1;
  int i,j,k;

  for(i = 0; i < p; i++){
    means[i] = 0.0;
    for(j = 0; j < p; j++){
      cov[j*p+i] = 0.0;
    }
  }

  for(i = 0; i < p; i++){
    for(j = 0; j < nSubset; j++){
      means[i] += x[(begin+j)*p+i];
    }
    means[i] = means[i]/nSubset;
  }


  for(i = 0; i < p; i++){
    for(j = i; j < p; j++){
      for(k = 0; k < nSubset; k++){
	cov[i*p+j] += (x[(begin+k)*p+i]-means[i])*(x[(begin+k)*p+j]-means[j]);
      }
      cov[i*p+j] = cov[i*p+j]/(nSubset-1);
    }
  }

}

double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}

void covTransInv(double *z, double *v, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      v[k] = z[k];
      if(i == j)
	v[k] = exp(z[k]);
    }
  }

}

void covTrans(double *v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTransInvExpand(double *v, double *z, int m){
  int i, j, k;

  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
      if(i == j)
	z[i*m+j] = exp(z[i*m+j]);
    }
  }

}

void covExpand(double *v, double *z, int m){
  int i, j, k;

  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
    }
  }

}

void printMtrx(double *m, int nRow, int nCol){

  int i, j;

  for(i = 0; i < nRow; i++){
    Rprintf("\t");
    for(j = 0; j < nCol; j++){
      Rprintf("%.3f\t", m[j*nRow+i]);
    }
    Rprintf("\n");
  }
}

void printVec(double *m, int n){

  Rprintf("\t");
    for(int j = 0; j < n; j++){
      Rprintf("%.3f\t", m[j]);
    }
    Rprintf("\n");
}

void printVec(int *m, int n){

  Rprintf(" ");
    for(int j = 0; j < n; j++){
      Rprintf("%i ", m[j]);
    }
    Rprintf("\n");
}

double logit_logpost(int &n, double *Y, double *eta, double *w){
  double loglike = 0.0;
  int i;

  for(i = 0; i < n; i++)
    loglike += Y[i]*(eta[i]+w[i]);

  for(i = 0; i < n; i++)
    loglike -= log(1.0+exp(eta[i]+w[i]));

  return loglike;
}

double binomial_logpost(int &n, double *Y, double *eta, double *w, int *weights){
  double p, loglike = 0.0;
  int i;

  for(i = 0; i < n; i++){
    p = 1.0/(1+exp(-eta[i]-w[i]));
    loglike += Y[i]*log(p)+(static_cast<double>(weights[i])-Y[i])*log(1.0-p);
  }

  return loglike;
}

double poisson_logpost(int &n, double *Y, double *eta, double *w, int *r){
  double loglike = 0.0;
  int i;

  for(i = 0; i < n; i++){
    loglike += Y[i]*(eta[i]+w[i]+log(static_cast<double>(r[i])))-exp(eta[i]+w[i]+log(static_cast<double>(r[i])));
  }

  return loglike;
}

double binomial_logpost(int &n, double *Y, double *eta, int *weights){
  double p, loglike = 0.0;
  int i;

  for(i = 0; i < n; i++){
    p = 1.0/(1+exp(-eta[i]));
    loglike += Y[i]*log(p)+(static_cast<double>(weights[i])-Y[i])*log(1.0-p);
  }

  return loglike;
}

double poisson_logpost(int &n, double *Y, double *eta, int *r){
  double loglike = 0.0;
  int i;

  for(i = 0; i < n; i++){
    loglike += Y[i]*(eta[i]+log(static_cast<double>(r[i])))-exp(eta[i]+log(static_cast<double>(r[i])));
  }

  return loglike;
}

void report(int &s, int &nSamples, int &status, int &nReport, bool &verbose){

  if(verbose){
    if(status == nReport){
      Rprintf("Sampled: %i of %i, %3.2f%%\n", s, nSamples, 100.0*s/nSamples);
      //Rprintf("---------------------------\n");
      #ifdef Win32
      R_FlushConsole();
      #endif
      status = 0;
    }
  }
  status++;

  R_CheckUserInterrupt();
}

void spCor(double *D, int n, double *theta, std::string &covModel, double *C){
  int i;

  if(covModel == "exponential"){

    for(i = 0; i < n; i++){
      C[i] = exp(-1.0*theta[0]*D[i]);
    }

  }else if(covModel == "spherical"){

    for(i = 0; i < n; i++){
      if(D[i] > 0 && D[i] <= 1.0/theta[0]){
	C[i] = 1.0 - 1.5*theta[0]*D[i] + 0.5*pow(theta[0]*D[i],3);
      }else if(D[i] >= 1.0/theta[0]){
	C[i] = 0.0;
      }else{
	C[i] = 1.0;
      }
    }

  }else if(covModel == "gaussian"){

    for(i = 0; i < n; i++){
      C[i] = exp(-1.0*(pow(theta[0]*D[i],2)));
    }

  }else if(covModel == "matern"){

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    for(i = 0; i < n; i++){
      if(D[i]*theta[0] > 0.0){
	C[i] = pow(D[i]*theta[0], theta[1])/(pow(2, theta[1]-1)*gammafn(theta[1]))*bessel_k(D[i]*theta[0], theta[1], 1.0);
      }else{
	C[i] = 1.0;
      }
    }

 }else{
    error("c++ error: cov.model is not correctly specified");
  }
}

double spCor(double D, double *theta, std::string &covModel){

  if(covModel == "exponential"){

    return exp(-1.0*theta[0]*D);

  }else if(covModel == "spherical"){

    if(D > 0 && D <= 1.0/theta[0]){
      return 1.0 - 1.5*theta[0]*D + 0.5*pow(theta[0]*D,3);
    }else if(D >= 1.0/theta[0]){
      return 0.0;
    }else{
      return 1.0;
    }

  }else if(covModel == "gaussian"){

    return exp(-1.0*(pow(theta[0]*D,2)));

  }else if(covModel == "matern"){

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    if(D*theta[0] > 0.0){
      return pow(D*theta[0], theta[1])/(pow(2, theta[1]-1)*gammafn(theta[1]))*bessel_k(D*theta[0], theta[1], 1.0);
    }else{
      return 1.0;
    }

  }else{
    error("c++ error: cov.model is not correctly specified");
    return 0;
  }
}

double spCor(double D, double phi, double nu, std::string &covModel){

  if(covModel == "exponential"){

    return exp(-1.0*phi*D);

  }else if(covModel == "spherical"){

    if(D > 0 && D <= 1.0/phi){
      return 1.0 - 1.5*phi*D + 0.5*pow(phi*D,3);
    }else if(D >= 1.0/phi){
      return 0.0;
    }else{
      return 1.0;
    }
  }else if(covModel == "matern"){

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    if(D*phi > 0.0){
      return pow(D*phi, nu)/(pow(2, nu-1)*gammafn(nu))*bessel_k(D*phi, nu, 1.0);
    }else{
      return 1.0;
    }
  }else if(covModel == "gaussian"){

    return exp(-1.0*(pow(phi*D,2)));

  }else{
    error("c++ error: cov.model is not correctly specified");
    return 0;
  }
}

void spCov(double *D, int n, double *theta, std::string &covModel, double *C){
  int i;

  if(covModel == "exponential"){

    for(i = 0; i < n; i++){
      C[i] = theta[0]*exp(-1.0*theta[1]*D[i]);
    }

  }else if(covModel == "spherical"){

    for(i = 0; i < n; i++){
      if(D[i] > 0 && D[i] <= 1.0/theta[1]){
	C[i] = theta[0]*(1.0 - 1.5*theta[1]*D[i] + 0.5*pow(theta[1]*D[i],3));
      }else if(D[i] >= 1.0/theta[1]){
	C[i] = 0.0;
      }else{
	C[i] = theta[0];
      }
    }

  }else if(covModel == "gaussian"){

    for(i = 0; i < n; i++){
      C[i] = theta[0]*exp(-1.0*(pow(theta[1]*D[i],2)));
    }

  }else if(covModel == "matern"){

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    for(i = 0; i < n; i++){
      if(D[i]*theta[1] > 0.0){
	C[i] = theta[0]*pow(D[i]*theta[1], theta[2])/(pow(2, theta[2]-1)*gammafn(theta[2]))*bessel_k(D[i]*theta[1], theta[2], 1.0);
      }else{
	C[i] = theta[0];
      }
    }

 }else{
    error("c++ error: cov.model is not correctly specified");
  }
}

void spCovLT(double *D, int n, double *theta, std::string &covModel, double *C){
  int i,j;

  if(covModel == "exponential"){

    for(i = 0; i < n; i++){
      for(j = i; j < n; j++){
	C[i*n+j] = theta[0]*exp(-1.0*theta[1]*D[i*n+j]);
      }
    }

  }else if(covModel == "spherical"){

    for(i = 0; i < n; i++){
      for(j = i; j < n; j++){
	if(D[i*n+j] > 0 && D[i*n+j] <= 1.0/theta[1]){
	  C[i*n+j] = theta[0]*(1.0 - 1.5*theta[1]*D[i*n+j] + 0.5*pow(theta[1]*D[i*n+j],3));
	}else if(D[i*n+j] >= 1.0/theta[1]){
	  C[i*n+j] = 0.0;
	}else{
	  C[i*n+j] = theta[0];
	}
      }
    }

  }else if(covModel == "gaussian"){

    for(i = 0; i < n; i++){
      for(j = i; j < n; j++){
	C[i*n+j] = theta[0]*exp(-1.0*(pow(theta[1]*D[i*n+j],2)));
      }
    }

  }else if(covModel == "matern"){

    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*pi/2*(besselI(d*phi,-nu)-besselI(d*phi, nu))/sin(nu*pi), or
    //(d*phi)^nu/(2^(nu-1)*gamma(nu))*besselK(x=d*phi, nu=nu)

    for(i = 0; i < n; i++){
      for(j = i; j < n; j++){
	if(D[i*n+j]*theta[1] > 0.0){
	  C[i*n+j] = theta[0]*pow(D[i*n+j]*theta[1], theta[2])/(pow(2, theta[2]-1)*gammafn(theta[2]))*bessel_k(D[i*n+j]*theta[1], theta[2], 1.0);
	}else{
	  C[i*n+j] = theta[0];
	}
      }
    }

 }else{
    error("c++ error: cov.model is not correctly specified");
  }
}

void transpose(double *m, int w, int h){
  int start, next, i;
  double tmp;

  for(start = 0; start <= w * h - 1; start++) {
    next = start;
    i = 0;
    do{	i++;
      next = (next % h) * w + next / h;
    }while (next > start);
    if(next < start || i == 1) continue;

    tmp = m[next = start];
    do{
      i = (next % h) * w + next / h;
      m[next] = (i == start) ? tmp : m[i];
      next = i;
    }while (next > start);
  }
}

void clearUT(double *m, int n){
  for(int i = 1; i < n; i++){
    for(int j = 0; j < i; j++){
      m[i*n+j] = 0;
    }
  }
}


// rwish delivers a pseudo-random Wishart deviate
//
// USAGE:
//
//   A <- rwish(v, S)
//
// INPUT:
//
//   v    degrees of freedom
//
//   S    Scale matrix
//
// OUTPUT:
//
//  A     a pseudo-random Wishart deviate
//
// Based on code originally posted by Bill Venables to S-news
// on 6/11/1998

// extern "C" {

//   SEXP rwish(SEXP S_r, SEXP v_r, SEXP p_r, SEXP Z_r, SEXP tmp_pp_r, SEXP iwish){

//     double *S = REAL(S_r);
//     int v = INTEGER(v_r)[0];
//     int p = INTEGER(p_r)[0];
//     double *Z = REAL(Z_r);
//     double *tmp_pp = REAL(tmp_pp_r);
//     bool riwish = static_cast<bool>(INTEGER(iwish));
void rwish(double *S, int v, int p, double *Z, double *tmp_pp, int iwish){

    int i, j, info;
    char const *lower = "L";
    char const *nUnit = "N";
    char const *ntran = "N";
    char const *ytran = "T";
    char const *rside = "R";
    const double one = 1.0;
    const double zero = 0.0;
    bool riwish = static_cast<bool>(iwish);

    if(riwish){
      F77_NAME(dpotrf)(lower, &p, S, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, S, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}
    }

    if(v < p){
      error("c++ error: rwish v < p\n");
    }

    F77_NAME(dpotrf)(lower, &p, S, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
    zeros(tmp_pp, p*p);

    //GetRNGstate();
    for(i = 0; i < p; i++){
      tmp_pp[i*p+i] = sqrt(rchisq(v-i));
    }

    for(j = 1; j < p; j++){
      for(i = 0; i < j; i++){
	tmp_pp[j*p+i] = rnorm(0, 1);
      }
    }
    //PutRNGstate();

    F77_NAME(dtrmm)(rside, lower, ytran, nUnit, &p, &p, &one, S, &p, tmp_pp, &p);
    F77_NAME(dgemm)(ytran, ntran, &p, &p, &p, &one, tmp_pp, &p, tmp_pp, &p, &zero, Z, &p);

    if(riwish){
      F77_NAME(dpotrf)(lower, &p, Z, &p, &info); if(info != 0){error("c++ error: dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &p, Z, &p, &info); if(info != 0){error("c++ error: dpotri failed\n");}
    }

    for(i = 1; i < p; i++){
      for(j = 0; j < i; j++){
	Z[i*p+j] = Z[j*p+i];
      }
    }

//     return(R_NilValue);
}
// }


