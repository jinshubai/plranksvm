#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include <omp.h>// include thie API of OPenMP
#include "linear.h"
#include "tron.h"
#include "selectiontree.h"

#ifdef FIGURE56
struct feature_node *x_spacetest;
struct problem probtest;
#endif
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
#ifdef FIGURE56
void evaluate_test(double* w)
{
	int i;
	double *true_labels = Malloc(double,probtest.l);
	double *dec_values = Malloc(double,probtest.l);
	if(&probtest != NULL)
	{
		for(i = 0; i < probtest.l; ++i)
		{
			feature_node *x = probtest.x[i];
			true_labels[i] = probtest.y[i];
			double predict_label = 0;
			for(; x->index != -1; ++x)
				predict_label += w[x->index-1]*x->value;
			dec_values[i] = predict_label;
		}
	}
	double result[3];
	eval_list(true_labels, dec_values, probtest.query, probtest.l, result);
	info("Pairwise Accuracy = %g%%\n",result[0]*100);
	info("MeanNDCG (LETOR) = %g\n",result[1]);
	info("NDCG (YAHOO) = %g\n",result[2]);
	free(true_labels);
	free(dec_values);
}
#endif

static int compare_id_and_value(const void *a, const void *b)//from large to small
{
	struct id_and_value *ia = (struct id_and_value *)a;
	struct id_and_value *ib = (struct id_and_value *)b;
	if(ia->value > ib->value)
		return -1;
	if(ia->value < ib->value)
		return 1;
	return 0;
}

class Reduce_Vectors
{
public:
	Reduce_Vectors(int size);
	~Reduce_Vectors();

	void init(void);
	void sum_scale_x(double scalar, feature_node *x);
	void reduce_sum(double* v);

private:
	int nr_thread;
	int size;
	double **tmp_array;
};

Reduce_Vectors::Reduce_Vectors(int size)
{
	nr_thread = omp_get_max_threads();
	this->size = size;
	tmp_array = new double*[nr_thread];
	for(int i = 0; i < nr_thread; i++)
		tmp_array[i] = new double[size];
}

Reduce_Vectors::~Reduce_Vectors(void)
{
	for(int i = 0; i < nr_thread; i++)
		delete[] tmp_array[i];
	delete[] tmp_array;
}

void Reduce_Vectors::init(void)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
		for(int j = 0; j < nr_thread; j++)
			tmp_array[j][i] = 0.0;
}

void Reduce_Vectors::sum_scale_x(double scalar, feature_node *x)
{
	int thread_id = omp_get_thread_num();

	while(x->index!=-1)
	{
		tmp_array[thread_id][x->index-1] += (scalar)*(x->value);
		x++;
	}
}

void Reduce_Vectors::reduce_sum(double* v)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
	{
		v[i] = 0;
		for(int j = 0; j < nr_thread; j++)
			v[i] += tmp_array[j][i];
	}
}

class selection_rank_fun: public function
{
	public:
		selection_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count);
		~selection_rank_fun();

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *s, double *Hs);

		int get_nr_variable(void);

	protected:
		void Xv(double *v, double *Xv);
		void XTv(double *v, double *XTv);

		double C;
		double *z;
		int *l_plus;
		int *l_minus;
		double *alpha_plus;
		double *alpha_minus;
		const problem *prob;
		int nr_subset;
		int *perm;
		int *start;
		int *count;
		id_and_value **pi;

		int *int_y;
		int *nr_class;

		Reduce_Vectors *reduce_vectors;

};

selection_rank_fun::selection_rank_fun(const problem *prob, double C, int nr_subset, int *perm, int *start, int *count)
{
	int i,j,k;

	int l=prob->l;
	this->prob = prob;
	this->nr_subset = nr_subset;
	this->perm = perm;
	this->start = start;
	this->count = count;
	this->C = C;
	l_plus = new int[l];
	l_minus = new int[l];
	alpha_plus = new double[l];
	alpha_minus = new double[l];
	z = new double[l];
	pi = new id_and_value* [nr_subset];

#pragma omp parallel for default(shared) private(i) if(nr_subset > 100)
	for (i=0;i<nr_subset;i++)
	{
		pi[i] = new id_and_value[count[i]];
	}

	reduce_vectors = new Reduce_Vectors(get_nr_variable());

	double *y=prob->y;
	int_y = new int[prob->l];
	nr_class = new int[nr_subset];
    
#pragma omp parallel for default(shared) private(i,j,k) 
	for (i=0;i<nr_subset;i++)
	{
		k=1;
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id=perm[j+start[i]];
			pi[i][j].value=y[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		int_y[pi[i][count[i]-1].id]=1;
		for(j=count[i]-2;j>=0;j--)
		{
			if (pi[i][j].value>pi[i][j+1].value)
				k++;
			int_y[pi[i][j].id]=k;
		}
		nr_class[i]=k;
	}
}

selection_rank_fun::~selection_rank_fun()
{
	int i;

    delete[] l_plus;
	delete[] l_minus;
	delete[] alpha_plus;
	delete[] alpha_minus;
	delete[] z;

#pragma omp parallel for default(shared) private(i) if(nr_subset > 100)
	for (i=0;i<nr_subset;i++)
		delete[] pi[i];
	delete[] pi;

	delete reduce_vectors;

	delete[] int_y;
	delete[] nr_class;
}

double selection_rank_fun::fun(double *w)
{
	int i,j,k;
	double reg = 0.0;
	double f=0.0;
	int l=prob->l;
	int w_size=get_nr_variable();
	selectiontree *T;
	Xv(w,z);

 #pragma omp parallel for default(shared) private(i, j, k, T) //construct the OST
	for (i=0;i<nr_subset;i++)
	{
		for (j=0;j<count[i];j++)
		{
			pi[i][j].id = perm[j+start[i]];
			pi[i][j].value = z[perm[j+start[i]]];
		}
		qsort(pi[i], count[i], sizeof(id_and_value), compare_id_and_value);

		T=new selectiontree(nr_class[i]);
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);
				k++;
			}
			T->count_smaller(int_y[pi[i][j].id],&l_minus[pi[i][j].id], &alpha_minus[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;

		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],pi[i][k].value);
				k--;
			}
			T->count_larger(int_y[pi[i][j].id],&l_plus[pi[i][j].id], &alpha_plus[pi[i][j].id]);
		}
		delete T;
	}
	long long nSV = 0;

#pragma omp parallel for default(shared) reduction(+:nSV) private(i)
	for (i=0;i<l;i++)
		nSV += l_plus[i];

	info("nSV = %ld\n",nSV);

//#pragma omp parallel default(shared)
{
	//#pragma omp parallel for private(i) reduction(+:reg) //if(w_size > 100)
	for(i=0;i<w_size;i++)
	{
		reg += w[i]*w[i];
	}

	//#pragma omp parallel for private(i) reduction(+:f)
	for(i=0;i<l;i++)
	{

		f += C*(z[i]*((l_plus[i]+l_minus[i])*z[i]-alpha_minus[i]-alpha_plus[i]-2*(l_minus[i]-l_plus[i]))+l_minus[i]);
	}
}

	return (reg/2.0+f);
}

void selection_rank_fun::grad(double *w, double *g)
{
	int i;
	int l=prob->l;
	double *ATAXw;
	ATAXw = new double[l];
	int w_size=get_nr_variable();

#pragma omp parallel for default(shared) private(i) 
	for (i=0;i<l;i++)
		ATAXw[i]=(double)l_plus[i]-(double)l_minus[i]+((double)l_plus[i]+(double)l_minus[i])*z[i]-alpha_plus[i]-alpha_minus[i];
	XTv(ATAXw, g);

#pragma omp parallel for default(shared) private(i) if(w_size > 100)
	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*C*g[i];
	delete[] ATAXw;
}

int selection_rank_fun::get_nr_variable(void)
{
	return prob->n;
}

void selection_rank_fun::Hv(double *s, double *Hs)
{
	int i,j,k;
	int w_size=get_nr_variable();
	int l=prob->l;
	double *wa = new double[l];
	selectiontree *T;
	double* alpha_plus_minus;
	alpha_plus_minus = new double[l];
	Xv(s, wa);

#pragma omp parallel for default(shared) private(i, j, k, T) 
	for (i=0;i<nr_subset;i++)
	{
		T=new selectiontree(nr_class[i]);// nr_class[i]is the i-th query's number of different label
		k=0;
		for (j=0;j<count[i];j++)
		{
			while (k<count[i]&&(1-pi[i][j].value+pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],wa[pi[i][k].id]);
				k++;
			}
			alpha_plus_minus[pi[i][j].id]=T->vector_sum_smaller(int_y[pi[i][j].id]);
		}
		delete T;
		k=count[i]-1;
		T = new selectiontree(nr_class[i]);
		for (j=count[i]-1;j>=0;j--)
		{
			while (k>=0&&(1+pi[i][j].value-pi[i][k].value>0))
			{
				T->insert_node(int_y[pi[i][k].id],wa[pi[i][k].id]);
				k--;
			}
			alpha_plus_minus[pi[i][j].id]+=T->vector_sum_larger(int_y[pi[i][j].id]);
		}
		delete T;
	}

#pragma omp parallel for default(shared) private(i)
	for (i=0;i<l;i++)
		wa[i]=wa[i]*((double)l_plus[i]+(double)l_minus[i])-alpha_plus_minus[i];
	delete[] alpha_plus_minus;
	XTv(wa, Hs);
	delete[] wa;

    #pragma omp parallel for default(shared) private(i) if(w_size > 100)
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*C*Hs[i];
}

void selection_rank_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

 #pragma omp parallel for default(shared) private(i) schedule(guided)
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void selection_rank_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	reduce_vectors->init();

#pragma omp parallel for default(shared) private(i) schedule(guided)
	for(i=0;i<l;i++)
		reduce_vectors->sum_scale_x(v[i], x[i]);
	
	reduce_vectors->reduce_sum(XTv);
}

	

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void group_queries(const problem *prob, int *nr_subset_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_subset = 16;
	int nr_subset = 0;
	int *query = Malloc(int,max_nr_subset);
	int *count = Malloc(int,max_nr_subset);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)prob->query[i];
		int j;
		for(j=0;j<nr_subset;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_subset)
		{
			if(nr_subset == max_nr_subset)
			{
				max_nr_subset *= 2;
				query = (int *)realloc(query,max_nr_subset*sizeof(int));
				count = (int *)realloc(count,max_nr_subset*sizeof(int));
			}
			query[nr_subset] = this_query;
			count[nr_subset] = 1;
			++nr_subset;
		}
	}

	int *start = Malloc(int,nr_subset);
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_subset;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_subset_ret = nr_subset;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn, int nr_subset=0, int *perm=NULL, int *start=NULL, int *count=NULL)
{
	double eps=param->eps;
	//clock_t begin,end;

	function *fun_obj=NULL;
	//begin = clock();
	switch(param->solver_type)
	{
		case SELECTION_TREE:
			{
				fun_obj=new selection_rank_fun(prob, param->C, nr_subset, perm, start, count);
				TRON tron_obj(fun_obj, param->eps);
				tron_obj.set_print_string(liblinear_print_string);
				tron_obj.tron(w);
				delete fun_obj;
				break;
			}
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
	//end = clock();
	//info("Training time = %g\n",double(end-begin)/double(CLOCKS_PER_SEC));
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	model_->nr_feature=n;
	model_->param = *param;
	
	if(param->solver_type == SELECTION_TREE)
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		int nr_subset;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);
		group_queries(prob, &nr_subset ,&start, &count, perm);
		train_one(prob, param, &model_->w[0],0,0, nr_subset, perm, start, count);
		free(start);
		free(count);
		free(perm);
	}
	else
	{
		fprintf(stderr, "Training cannot be executed because of unknow solver_type.");
		exit(1);
	}
	return model_;
}

static void group_queries(const int *query_id, int l, int *nr_query_ret, int **start_ret, int **count_ret, int *perm)
{
	int max_nr_query = 16;
	int nr_query = 0;
	int *query = Malloc(int,max_nr_query);
	int *count = Malloc(int,max_nr_query);
	int *data_query = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_query = (int)query_id[i];
		int j;
		for(j=0;j<nr_query;j++)
		{
			if(this_query == query[j])
			{
				++count[j];
				break;
			}
		}
		data_query[i] = j;
		if(j == nr_query)
		{
			if(nr_query == max_nr_query)
			{
				max_nr_query *= 2;
				query = (int *)realloc(query,max_nr_query * sizeof(int));
				count = (int *)realloc(count,max_nr_query * sizeof(int));
			}
			query[nr_query] = this_query;
			count[nr_query] = 1;
			++nr_query;
		}
	}

	int *start = Malloc(int,nr_query);
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_query[i]]] = i;
		++start[data_query[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_query;i++)
		start[i] = start[i-1] + count[i-1];

	*nr_query_ret = nr_query;
	*start_ret = start;
	*count_ret = count;
	free(query);
	free(data_query);
}

void eval_list(double *label, double *target, int *query, int l, double *result_ret)
{
	int q,i,j,k;
	int nr_query;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int, l);
	id_and_value *order_perm;
	int true_query;
	int ndcg_size;
	long long totalnc = 0, totalnd = 0;
	long long nc = 0;
	long long nd = 0;
	double tmp;
	double accuracy = 0;
	int *l_plus;
	int *int_y;
	int same_y = 0;
	double *ideal_dcg;
	double *dcg;
	double meanndcg = 0;
	double ndcg;
	double dcg_yahoo,idcg_yahoo,ndcg_yahoo;
	selectiontree *T;
	group_queries(query, l, &nr_query, &start, &count, perm);
	true_query = nr_query;
	for (q=0;q<nr_query;q++)
	{
		//We use selection trees to compute pairwise accuracy
		nc = 0;
		nd = 0;
		l_plus = new int[count[q]];
		int_y = new int[count[q]];
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		int_y[order_perm[count[q]-1].id] = 1;
		same_y = 0;
		k = 1;
		for(i=count[q]-2;i>=0;i--)
		{
			if (order_perm[i].value != order_perm[i+1].value)
			{
				same_y = 0;
				k++;
			}
			else
				same_y++;
			int_y[order_perm[i].id] = k;
			nc += (count[q]-1 - i - same_y);
		}
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = i;
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		//total pairs
		T = new selectiontree(k);
		j = count[q] - 1;
		for (i=count[q] - 1;i>=0;i--)
		{
			while (j>=0 && ( order_perm[j].value < order_perm[i].value))
			{
				T->insert_node(int_y[order_perm[j].id], tmp);
				j--;
			}
			T->count_larger(int_y[order_perm[i].id], &l_plus[order_perm[i].id], &tmp);
		}
		delete T;

		for (i=0;i<count[q];i++)
			nd += l_plus[i];
		nc -= nd;
		if (nc != 0 || nd != 0)
			accuracy += double(nc)/double(nc+nd);
		else
			true_query--;
		totalnc += nc;
		totalnd += nd;
		delete[] l_plus;
		delete[] int_y;
		delete[] order_perm;
	}
	result_ret[0] = (double)totalnc/(double)(totalnc+totalnd);
	for (q=0;q<nr_query;q++)
	{
		ndcg_size = min(10,count[q]);
		ideal_dcg = new double[count[q]];
		dcg = new double[count[q]];
		ndcg = 0;
		order_perm = new id_and_value[count[q]];
		int *perm_q = &perm[start[q]];
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = label[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		ideal_dcg[0] = pow(2.0,order_perm[0].value) - 1;
		idcg_yahoo = pow(2.0, order_perm[0].value) - 1;
		for (i=1;i<count[q];i++)
			ideal_dcg[i] = ideal_dcg[i-1] + (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+1.0);
		for (i=1;i<ndcg_size;i++)
			idcg_yahoo += (pow(2.0,order_perm[i].value) - 1) * log(2.0) / log(i+2.0);
		for (i=0;i<count[q];i++)
		{
			order_perm[i].id = perm_q[i];
			order_perm[i].value = target[perm_q[i]];
		}
		qsort(order_perm, count[q], sizeof(id_and_value), compare_id_and_value);
		dcg[0] = pow(2.0, label[order_perm[0].id]) - 1;
		dcg_yahoo = pow(2.0, label[order_perm[0].id]) - 1;
		for (i=1;i<count[q];i++)
			dcg[i] = dcg[i-1] + (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 1.0);
		for (i=1;i<ndcg_size;i++)
			dcg_yahoo += (pow(2.0, label[order_perm[i].id]) - 1) * log(2.0) / log(i + 2.0);
		if (ideal_dcg[0]>0)
			for (i=0;i<count[q];i++)
				ndcg += dcg[i]/ideal_dcg[i];
		else
			ndcg = 0;
		meanndcg += ndcg/count[q];
		delete[] order_perm;
		delete[] ideal_dcg;
		delete[] dcg;
		if (idcg_yahoo > 0)
			ndcg_yahoo += dcg_yahoo/idcg_yahoo;
		else
			ndcg_yahoo += 1;
	}
	meanndcg /= nr_query;
	ndcg_yahoo /= nr_query;
	result_ret[1] = meanndcg;
	result_ret[2] = ndcg_yahoo;
	free(start);
	free(count);
	free(perm);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}
	return dec_values[0];
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

static const char *solver_type_table[]={"SELECTION_TREE",NULL};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");

				setlocale(LC_ALL, old_locale);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != SELECTION_TREE)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

