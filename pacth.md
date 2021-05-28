\documentclass{article}

% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
% before loading neurips_2021

% ready for submission
\usepackage{neurips_2021}

% to compile a preprint version, e.g., for submission to arXiv, add add the
% [preprint] option:
%     \usepackage[preprint]{neurips_2021}

% to compile a camera-ready version, add the [final] option, e.g.:
%     \usepackage[final]{neurips_2021}

% to avoid loading the natbib package, add option nonatbib:
%    \usepackage[nonatbib]{neurips_2021}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{wrapfig}
\usepackage{subfigure}

\newcommand{\R}{\mathbb{R}}
\newcommand{\I}{\mathbb{I}}
\newcommand{\calL}{\mathcal{L}}
\newcommand{\calC}{\mathcal{C}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\KL}{D_{KL}}
\newcommand{\calN}{\mathcal{N}}

\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}


\title{Towards Robust Active Feature Acquisition}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.

\author{%
  David S.~Hippocampus\thanks{Use footnote for providing further information
    about author (webpage, alternative address)---\emph{not} for acknowledging
    funding agencies.} \\
  Department of Computer Science\\
  Cranberry-Lemon University\\
  Pittsburgh, PA 15213 \\
  \texttt{hippo@cs.cranberry-lemon.edu} \\
  % examples of more authors
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \AND
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
}

\begin{document}

\maketitle

\begin{abstract}
Truly intelligent systems are expected to make critical decisions with incomplete and uncertain data. Active feature acquisition (AFA), where features are sequentially acquired to improve the prediction, is a step towards this goal. However, current AFA models all deal with a small set of candidate features and have difficulty scaling to a large feature space. Moreover, they are ignorant about the valid domains where they can predict confidently, thus they can be vulnerable to out-of-distribution (OOD) inputs. In order to remedy these deficiencies and bring AFA models closer to practical use, we propose several techniques to advance the current AFA approaches. Our framework can easily handle a large number of features using a hierarchical acquisition policy and is more robust to OOD inputs with the help of an OOD detector for partially observed data. Extensive experiments demonstrate the efficacy of our framework over strong baselines.
\end{abstract}

\section{Introduction}
A typical machine learning system will first collect all the features and then predict the target variables based on the collected feature values. Unlike the two-step paradigm, active feature acquisition performs feature value acquisition and target prediction at the same time. Features are actively acquired to improve the prediction, and the prediction in turn informs the acquisition process. Ideally, only features that provide unique information and outweigh their cost will be acquired. The AFA model will stop acquiring more features when the prediction is sufficiently accurate or it exceeds the given acquisition budget. Since each instance could have different set of informative features, active feature acquisition is expected to acquire different features for different instances.

As a motivating example, consider a doctor making a diagnosis on a patient (an instance). The doctor usually has not observed all the possible measurements (such as blood samples, x-rays, etc.) from the patient. The doctor is also not forced to make a diagnosis based on the currently observed measurements; instead, he/she may dynamically decide to take more measurements to help determine the diagnosis. The next measurement to make (feature to observe), if any, will depend on the values of the already observed features; thus, the doctor may determine a different set of features to observe from patient to patient (instance to instance) depending on the values of the features that were observed. Hence, each patient will not have the same subset of features selected (as would be the case with typical feature selection).

In the current literature, there are mainly two types of approaches to acquire features actively: greedy acquisition approaches and reinforcement learning based approaches. Both approaches acquire features sequentially, that is, one candidate feature is acquired at each acquisition step based on the previously observed features. Greedy approaches directly optimize the utility of the next acquisition, while reinforcement learning based approaches optimize a discounted reward along the acquisition trajectories. As a result, the reinforcement learning (RL) based approaches tend to find better solutions to the AFA problem as shown in \cite{li2020active}. Here, we base ourselves on the Markov decision process (MDP) formulation of the AFA problem proposed in \cite{li2020active,shim2018joint} and focus on resolving the deficiencies of the current AFA models.

One of the obstacles to extending the current AFA models to practical use is the potentially large number of candidate features. Greedy approaches are computationally difficult to scale, because the utilities need to be recalculated for each candidate feature based on the updated set of observed features at each acquisition step, which incurs an $O(d^2)$ complexity for a $d$ dimensional feature space. Reinforcement learning algorithms are known to have difficulties with a high dimensional action space \cite{dulac2015deep}. In this work, we propose to cluster the candidate features into groups and use a hierarchical reinforcement learning agent to select the next feature to be acquired.

Another challenge in deploying the AFA models is their robustness. In a practical application, it is very likely for an AFA model to encounter inputs that are different from its training distribution. For example, it may be asked to acquire features for patients with unknown disease. For those out-of-distribution instances, the AFA model may acquire an arbitrary subset of features and predict one of its known categories. Dealing with out-of-distribution inputs is difficult in general, and it is even more challenging for AFA models, since the model only has access to a subset of features at any acquisition step. In this work, we propose a novel algorithm for detecting out-of-distribution inputs with partially observed features, and further utilize it to improve the robustness of the AFA model.

Our contributions are as follows: 1) We propose to reduce the action space for active feature acquisition by grouping similar actions and further learn a hierarchical policy to select the next candidate feature to be acquired. 2) We develop a novel out-of-distribution detection algorithm that can distinguish OOD inputs using an arbitrary subset of features. 3) Armed with the partially observed OOD detection algorithm, we encourage the AFA agent to acquire features that are most informative for distinguishing OOD inputs. 4) Our approach achieves the state-of-the-art performance for active feature acquisition, meanwhile it is more robust to out-of-distribution inputs.

\section{Background}\label{sec:background}
\subsection{Active Feature Acquisition (AFA)}\label{sec:afa}
Typical discriminative models predict a target $y$ using all $d$-dimensional features $x\in\R^d$. AFA, instead, actively acquires feature values to improve the prediction. It typically starts from an empty set of features and sequentially acquires more features $x_{i}$ until the prediction is sufficiently accurate or it exceeds the given acquisition budget. The goal of an AFA model is to minimize the following objective
\begin{equation}
    \calL(\hat{y}(x_{o}),y) + \alpha \calC(o),
\label{eq:objetive}
\end{equation}
where $\calL(\hat{y}(x_{o}),y)$ is the prediction error between the groundtruth target $y$ and the prediction $\hat{y}$ using the acquired features $x_o$, $\calC(o)$ measures the total cost of acquiring a subset of features $o \subseteq \{1,\ldots,d\}$, and $\alpha$ balances these two terms.

However, directly optimizing \eqref{eq:objetive} is not trivial, since it involves optimizing over a combinatorial number of possible subsets. Many heuristic approaches have been developed to approximately solve this problem. For example, in \cite{ling2004decision}, the authors propose to take into account the cost of features when selecting an attribute for building a decision tree so that the final tree will have a minimum total cost. \cite{chai2004test} utilizes a naive Bayes classifier to handle the partially observed features, where the unobserved features are simply ignored in the likelihood objective. They then assess the utility of each unobserved feature by their expected reduction of the misclassification cost. At each acquisition step, the feature with highest utility is acquired. Authors in \cite{nan2014fast} instead leverage a margin-based classifier. An instance is classified by retrieving the nearest neighbors from training set using the partially observed features, and the utility of each unobserved feature is calculated by the one-step ahead classification accuracy. Following the same greedy solution, EDDI \cite{ma2018eddi} utilizes the modern generative models to handle partially observed instances. Specifically, they propose a partial VAE to model the arbitrary marginal likelihoods $p(x_o)$ (target variable $y$ is concatenated into $x$ and modeled together). Inspired by the experimental design approaches \cite{bernardo1979expected}, they assess the utility of each unobserved feature with their expected information gain to the target variable $y$, i.e.,
\begin{equation}
    \mathcal{U}_i = \E_{p(x_i \mid x_o)} \KL[p(y \mid x_o, x_i) \| p(y \mid x_o)].
\end{equation}
The feature with highest utility is acquired at each step. Similar to EDDI, Icebreaker \cite{gong2019icebreaker} proposes to use a Bayesian Deep Latent Gaussian model to capture the uncertainty of unobserved features and to assess their utilities. It further extend the problem to actively acquire additional information during training.

Greedy approaches are easy to understand, but they are also inherently flawed, since they are myopic and unaware of the long-term goal of obtaining multiple features that are \emph{jointly} informative. Instead of acquiring features greedily, the AFA problem has been formulated as a Markov Decision Process (MDP) \cite{zubek2004pruning,ruckstiess2011sequential}. Therefore, reinforcement learning based approaches can be utilized, where a long-term discounted reward is optimized. In the MDP formulation of AFA problem, the state is the current observed features, the action is the next feature to acquire, and the reward contains the final prediction reward and the cost of each acquired feature. In \cite{li2020active} and \cite{shim2018joint}, a special action indicating the termination of the acquisition process is also introduced. The agent will stop acquiring more features when it selects the termination action. Specifically, we have
\begin{equation}
    s = [o, x_o], \qquad a \in u \cup \phi, \qquad r(s,a)=-\calL(\hat{y}(x_{o}),y)\I(a=\phi)-\alpha \calC(a)\I(a \neq \phi),
\label{eq:mdp}
\end{equation}
where the state, $s$, consists of the current acquired feature subset, $o \subseteq \{1,\ldots,d\}$, and their values, $x_o$. The action, $a$, is either one of the remaining unobserved features, $u=\{1, \ldots, d\} \setminus o$, or the termination action, $\phi$. When a new feature, $i$, is acquired, the current state transits to a new state following $o \xrightarrow{i} o \cup i$, $x_o \xrightarrow{i} x_o \cup x_i$, and the agent receives the negative acquisition cost of this feature as a reward. If the termination action is selected (i.e., $a = \phi$), the agent makes a prediction based on all acquired features, $x_o$, and receives a final reward as $-\calL(\hat{y}(x_{o}),y)$.

Given the above MDP formulation, several RL approaches have been explored. \cite{zubek2004pruning} fits a transition model using complete data, and then uses the $\text{AO}^*$ heuristic search algorithm to find an optimal policy. \cite{ruckstiess2011sequential} utilizes Fitted Q-Iteration to optimize the MDP. \cite{he2012imitation} and \cite{he2016active} instead employ an imitation learning approach coached by a greedy reference policy. JAFA \cite{shim2018joint} jointly learns an RL agent and a classifier, where the classifier is deemed as the environment to calculate the classification error as the reward. 

Although MDPs are broad enough to encapsulate the active acquisition of features, there are several challenges that limit the success of a naive reinforcement learning approach. In the aforementioned MDP, the agent pays the acquisition cost at each acquisition step but only receives a reward about the prediction after completing the acquisition process. This results in sparse rewards leading to credit assignment problems for potentially long episodes \cite{minsky1961steps,sutton1988learning}, which may make training difficult. In addition, an agent that is making feature acquisitions must also navigate a complicated high-dimensional action space, as the action space scales with the number of features, making for a challenging RL problem \cite{dulac2015deep}. Finally, the agent must manage multiple roles as it has to: implicitly model dependencies (to avoid selecting redundant features); perform a cost/benefit analysis (to judge what unobserved informative feature is worth the cost); and act as a classifier (to make a final prediction). Attaining these roles individually is a challenge, doing so jointly and without gradients on the reward (as with the MDP) is an even more daunting problem. To lessen the burden on the agent, and assuage the other challenges, GSMRL \cite{li2020acflow} proposes a model-based alternative. The key observation of GSMRL is that the dynamics of the above MDP can be modeled by the conditional dependencies among features.
% the arbitrary conditional distributions $p(x_u \mid x_o)$, where $u, o \subseteq \{1,\ldots,d\}$. 
That is, the state transitions are based on the conditionals: $p(x_j \mid x_o)$, where $x_o$ is current observed features and $x_j$ is an unobserved feature. Base on this observation, a surrogate model that captures the arbitrary conditionals, $p(x_u, y \mid x_o)$, is employed to assist the agent. 
% Since the arbitrary conditionals capture the dependencies among subset of features \cite{li2020acflow}, they can be utilized to assess the informativeness of each unobserved feature. 
Specifically, GSMRL defines an intermediate reward using the information gain of the acquired feature, $x_i$, at the current acquisition step, i.e.,
$
    r_m(s,i) = H(y \mid x_o) -  H(y \mid x_o, x_i).
$
The entropy terms are estimated using the learned surrogate model.
Furthermore, the expected information gain for each candidate acquisition is provided to the agent as side information, i.e.,
\begin{equation}
\mathcal{U}_j = H(y \mid x_o) - \E_{p(x_j \mid x_o)}H(y \mid x_o, x_j), \quad j \in u.
\end{equation}
In addition to $\mathcal{U}_j$, the surrogate model can also provide the current prediction $\hat{y}$, the prediction probability, $p(y \mid x_o)$, the imputed values of unobserved features and their uncertainties, $p(x_u \mid x_o)$, as auxiliary information. Armed with the auxiliary information and the intermediate reward, GSMRL alleviates the challenge of a model-free approach and obtains state-of-the-art performance for several AFA problems. Given their established excellence, we use GSMRL as the base model for our robust AFA framework.

\subsection{Active Instance Recognition (AIR)}
AFA acquires features actively to improve the prediction of a target variable, while some application do not have an explicit target; instead, the features are acquired to improve our understanding to the instance. In GSMRL \cite{li2020active}, the authors propose a task named AIR, where an agent acquires features actively to reconstruct the unobserved features. A similar model-based RL approach is used for AIR, where a dynamics model $p(x_u \mid x_o)$ captures the state transition. The intermediate reward and auxiliary information can be similarly derived by replacing $y$ with $x_{u \setminus i}$ (the unobserved features excluding the current candidate $i$). A special case of AIR is to acquire features in k-space for accelerated MRI reconstruction. \cite{pineda2020active} and \cite{bakker2020experimental} have explored this application using deep Q-learning and policy gradient respectively. Several non-RL approaches \cite{zhang2019reducing,gorp2021active} have also been proposed.

\subsection{Out-of-distribution Detection}\label{sec:ood}
ML models are typically trained with a specific data distribution, however, when deployed, the model may encounter data that is outside the training distribution. For those out-of-distribution inputs, the prediction could be arbitrarily bad. Therefore, detecting OOD inputs has been an active research direction. One approach relies on the uncertainty of prediction. The prediction for OOD inputs are expected to have higher uncertainty. However, deep classifiers tend to be overly confident about their prediction, thus a postprocessing step to calibrate the uncertainty is needed \cite{ovadia2019can,kumar2019verified}. Bayesian neural networks (BNNs), where weights are sampled from their posterior distributions, have also been used to quantify the uncertainty \cite{blundell2015weight}. However, the full posterior is usually intractable. To avoid the complex posterior inference of BNNs, deep ensemble \cite{lakshminarayanan2016simple} and MC dropout \cite{gal2016dropout} are proposed as two approximations, where multiple independent forward passes are conducted to obtain a set of predictions. To avoid the expensive multiple forward passes, DUQ \cite{van2020uncertainty} propose to detect OOD inputs with a single deterministic model in a single pass. It learns a set of centroid vectors corresponding to the different classes and measures uncertainty according to the distance calculated by an RBF network \cite{lecun1998gradient} between the model output and its closest centroid. Recently, Gaussian processes have been combined with deep neural networks to quantify the prediction uncertainty, such as SNGP \cite{liu2020simple} and DUE \cite{van2021improving}.

Intuitively, generative models are expected to be able to detect OOD inputs using the likelihood \cite{bishop1994novelty}, that is, OOD inputs should have lower likelihood than the in-distribution ones. However, recent works \cite{nalisnick2018do, hendrycks2018deep} show that it is not the case for high dimensional distributions. This oddity has been verified for many modern deep generative models, such as PixelCNN, VAE, and normalizing flows. To rectify this pathology, \cite{choi2018waic} proposes to train multiple generative models in parallel and utilize the Watanabe-Akaike Information Criterion to identify OOD data. \cite{ren2019likelihood} proposes to use likelihood ratio of an autoregressive model to remove the contributions from background pixels. \cite{nalisnick2019detecting} propose a simple typicality test for a batch of data.
Furthermore, \cite{morningstar2021density} propose to utilize an estimator for the density of states on several summary statistics of the in-distribution data. It then evaluates the density of states estimator (DoSE) on testing data and marks those with low support as OOD. MSMA \cite{mahmood2021multiscale} empirically observes that the norm of scores conditioned on different noise levels serves as very effective summary statistics in the DoSE framework, and thus utilizes the multi-scale score matching network \cite{song2019generative} to calculate those statistics.

\section{Method}
In this section, we introduce each component of our framework. We use the model-based AFA approach, GSMRL \cite{li2020active}, as the base model, which utilizes an arbitrary conditional model $p(x_u, y \mid x_o)$ to assist the agent by providing the intermediate rewards and the auxiliary information. We further leverage the arbitrary conditionals to cluster features into groups and develop a hierarchical acquisition policy to deal with the large action space. Next, we introduce the OOD detection algorithm for partially observed instances along the acquisition trajectories. We then compose all those components together and propose the robust active feature acquisition framework. For convenience of evaluating OOD detection performance, we do not use the termination action for AFA but specify a budget of the acquisition (i.e., the number of features being acquired).

\subsection{Action Space Grouping}
As described in Sec.~\ref{sec:afa}, the AFA problem can be interpreted as a MDP, where the action space at each acquisition step contains the current unobserved features. For certain problems, the action space could be enormous. For example, in the aforementioned health care example, the action space could contain an exhaustive list of possible inspections a hospital can offer. Dealing with large action space for RL is generally challenging, since the agent may not be able to effectively explore the entire action space. Several approaches have been proposed to train RL agent with a large discrete action space. For instance, \cite{dulac2015deep} proposes a Wolpertinger policy that maps a state representation to a continuous action representation. The action representation is then used to look up $k$-nearest valid actions as candidates. Finally, the action with the highest Q value is selected and executed in the environment. \cite{majeed2020exact} proposes a sequentialization scheme, where the action space is transformed into a sequence of $\mathcal{B}$-ary decision code words. A pair of bijective encoder-decoder is defined to perform this transformation. Running the agent will produce a sequence of decisions, which are subsequently decoded as a valid action that can be executed in the environment.

Similar to \cite{majeed2020exact}, we also formulate the action space as a sequence of decisions. Here, we propose to utilize the inherited clustering properties of the candidate features. Given a set of features, $\{x_1,\ldots,x_d\}$, we assume features can be clustered based on their informativeness to the target variable $y$. That is, there might be a subset of features that are decisive about $y$ and another subset of features that are not relevant to $y$. This assumption holds true for many real-life tasks. For example, a music recommender system might use a questionnaire to collect information from a user. Some questions about musicians or song genres are closely related to the target, while some questions about addresses might not be relevant at all. Based on this intuition, we propose to assess the informativeness of the candidate features using their mutual information to the target variable, $y$, i.e., $I(x_i ; y)$, where $i \in \{1,\ldots,d\}$. The mutual information can be estimated using the learned arbitrary conditionals of the surrogate model
\begin{equation}
    I(x_i; y) = \E_{x_i,y} \log \frac{p(x_i, y)}{p(x_i)p(y)} = \E_{x_i,y} \log \frac{p(y \mid x_i)}{p(y \mid \emptyset)},
\end{equation}
where the expectation is estimated using a held-out validation set. Given the estimated mutual information, we can simply sort and divide the candidate features into different groups. For the sake of implementation simplicity, we use clusters with the same number of features. We can further group features inside each cluster into smaller clusters and develop a tree-structured action space as in \cite{majeed2020exact}, which we leave for future works. Note that the clustering is not performed actively for each instance; instead, we cluster once for each dataset and keep the cluster structure fixed throughout the acquisition process.

It is worth noting that the mutual information $I(x_i ; y)$ is not the only choice for clustering features. Alternative quantities, such as the mutual information $I(x_i ; x_j)$ or a metric $d(x_i, x_j) = H(x_i, x_j) - I(x_i ; x_j)$, can be used together with a hierarchical clustering procedure to group candidate features. However, these alternatives need to be estimated for each pair of candidate features, which incurs a $O(d^2)$ computational complexity, while the mutual information, $I(x_i ; y)$, only has $O(d)$ complexity.

Given the grouped action space, $\mathcal{A} = \{g_k\}_{k=1}^{K}$, with $K$ distinct clusters, we develop a hierarchical policy to select one candidate feature at each acquisition step. $g_k = \{g_k^{(1)}, \ldots, g_k^{(N)}\} \subseteq \{1,\ldots,d\}$ represents the $k_{th}$ group of features of size $N$, where $\forall k \neq k',\ g_k \cap g_{k'} = \emptyset$ and $\cup_{k=1}^K g_k = \{1, \ldots, d\}$. The policy factorizes autoregressively by first selecting the group index, $k$, and then selecting the feature index, $n$, inside the selected group, i.e.,
\begin{equation}\label{eq:autoreg_action}
    p(a \mid s) = p(k \mid s)p(n \mid k, s), \quad k \in \{1, \ldots, K\}, \quad n \in \{1, \ldots, N\}.
\end{equation}
The actual feature index being acquired is then decoded as $g_k^{(n)}$. As the agent acquires features, the already acquired features are removed from the candidate set. We simply set the probabilities of those features to zeros and renormalize the distribution. Similarly, if all features of a group have been acquired, the probability of this group is set to zero. With the proposed action space grouping, the original $d$-dimensional action space is reduced to $K+N$ decisions. Please refer to Fig.~\ref{fig:action_space} for an illustration.

\begin{figure}
\begin{minipage}{0.47\linewidth}
    \centering
    \includegraphics[width=0.87\linewidth]{imgs/action.png}
    \caption{An illustrative example of the grouped action space, where 6 features are grouped into 3 clusters. The grayed circles represent the current observed features (or fully observed groups) and are not considered as candidates anymore. The dashed line shows one acquisition at the current step, which acquires the feature $g_2^{(1)}$. The corresponding circles will be grayed after this acquisition step.}
    \label{fig:action_space}
\end{minipage}
\hfill
\begin{minipage}{0.48\linewidth}
\begin{algorithm}[H]
\caption{Robust Active Feature Acquisition}
\label{alg:robust_afa}
\begin{algorithmic}
\REQUIRE{acquisition environment $\textit{env}$; dynamics model $\textit{M}$; partially observed OOD detector $\textit{D}$; AFA agent $\textit{agent}$; acquisition budget $\textit{B}$}
\ENSURE{reward, ood\_likelihood, prediction}
\STATE{$x_o$, o, reward = \textit{env}.reset()}
\WHILE{$|o| < \textit{B}$}{
  \STATE{aux = $\textit{M}$.query($x_o$, o)}
  \STATE{action = \textit{agent}.act($x_o$, o, aux)}
  \STATE{$r_m =$ $\textit{M}$.reward($x_o$, $o$, action)}
  \STATE{$x_o$, o, $r_e$ = \textit{env}.step(action)}
  \STATE{reward += $r_e$ + $r_m$}}
\ENDWHILE
\STATE{aux = $\textit{M}$.query($x_o$, o)}
\STATE{prediction = $\textit{agent}$.predict($x_o$, o, aux)}
\STATE{$r_p = $ $\textit{env}$.reward($x_o$, $o$, prediction)}
\STATE{$r_d =$ $\textit{D}$.reward($x_o$, $o$)}
\STATE{reward += $r_p$ + $r_d$}
\STATE{ood\_likelihood = $\textit{D}$.log\_prob($x_o$, o)}
\end{algorithmic}
\end{algorithm}
\end{minipage}
\end{figure}

\subsection{Partially Observed Out-of-distribution Detection}
In Sec.~\ref{sec:ood}, we introduce several advanced techniques to detect out-of-distribution inputs. However, those approaches require fully observed data. In an AFA framework, data are partially observed at any acquisition step, which renders those approaches inappropriate. In this section, we develop a novel OOD detection algorithm specifically tailored for partially observed data. 
% Our approach extends MSMA \cite{mahmood2021multiscale}, the state-of-the-art OOD detection algorithm, to partially observed inputs.
Inspired by MSMA \cite{mahmood2021multiscale}, we propose to use the norm of scores from an arbitrary marginal distribution $p(x_o)$ as summary statistics and further detect partially observed OOD inputs with a DoSE \cite{morningstar2021density} approach.
MSMA for fully observed data is built by the following steps:
\begin{enumerate}[label=(\roman*)]
    \item Train a noise conditioned score matching network $s_\theta$ \cite{song2019generative} with $L$ noise levels by optimizing
    \begin{equation}\label{eq:score_matching}
        \frac{1}{L}\sum_{i=1}^{L} \frac{\sigma_i^2}{2} \E_{p_{data}(x)}\E_{\tilde{x} \sim \calN(x, \sigma_i^2 I)} \left[ \left\Vert s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x}-x}{\sigma_i^2} \right\Vert_2^2 \right].
    \end{equation}
    The score network essentially approximates the score of a series of smoothed data distributions $\nabla_{\tilde{x}} \log q_{\sigma_i}(\tilde{x})$, where $q_{\sigma_i}(\tilde{x}) = \int p_{data}(x)q_{\sigma_i}(\tilde{x} \mid x) dx$, and $q_{\sigma_i}(\tilde{x} \mid x)$ transforms $x$ by adding some Gaussian noise form $\calN(0, \sigma_i^2I)$.
    
    \item For a given input $x$, compute the L2 norm of scores at each noise level, i.e., $s_i = \Vert s_\theta(x, \sigma_i) \Vert$.
    
    \item Fit a low dimensional likelihood model for the norm of scores using in-distribution data, i.e., $p(s_1, \ldots, s_L)$, which is called density of states in \cite{morningstar2021density} following the concept in statistical mechanics.
    
    \item Threshold the likelihood to determine whether the input $x$ is OOD or not.
\end{enumerate}
In order to deal with partially observed data, we modify the score network to output scores of arbitrary marginal distributions, i.e., $\nabla_{\tilde{x}_m}\log q_{\sigma_i}(\tilde{x}_m)$, where $m \subseteq \{1, \ldots, d\}$ represents an arbitrary subset of features. The training objective \eqref{eq:score_matching} is modified accordingly to
\begin{equation}
    \frac{1}{L}\sum_{i=1}^{L} \frac{\sigma_i^2}{2} \E_{p_{data}(x)}\E_{\tilde{x} \sim \calN(x, \sigma_i^2 I)}\E_{m \sim p(m)} \left[ \left\Vert s_\theta(\tilde{x} \odot \I_m, \I_m, \sigma_i) \odot \I_m + \frac{\tilde{x} \odot \I_m-x \odot \I_m}{\sigma_i^2} \right\Vert_2^2 \right],
\end{equation}
where $\I_m$ represents a $d$-dimensional binary mask indicating the partially observed features, $\odot$ represents the element-wise product operation, and $p(m)$ is the distribution for generating observed dimensions. Similar to the fully observed case, we compute the L2 norm of scores at each noise level, i.e., $s_i = \Vert s_\theta(x \odot \I_m, \I_m, \sigma_i) \odot \I_m \Vert$, and fit a likelihood model in this transformed low-dimensional space. The likelihood model is also conditioned on the binary mask $\I_m$ to indicate the observed dimensions, i.e., $p(s_1,\ldots,s_L \mid \I_m)$. Given an input $x$ with observed dimensions $m$, we threshold the likelihood $p(s_i, \ldots, s_L \mid \I_m)$ to determine whether the partially observed data $x_m$ is OOD or not. 
To train the partially observed MSMA (PO-MSMA), we generate a mask for each input data $x$ at random. The conditional likelihood over norm of scores is estimated by a conditional autoregressive model, for which we utilize the efficient masked autoregressive implementation \cite{papamakarios2017masked}.

One benefit of our proposed PO-MSMA approach is that a single model can be used to detect OOD inputs with arbitrary observed features, which is convenient for detecting OOD inputs along the acquisition trajectories. Furthermore, sharing weights across different tasks (i.e., different marginal distributions) could act as a regularization (as discussed in \cite{li2020acflow}), thus the unified score matching network can potentially perform better than separately trained ones for each different conditional, which we will investigate in future works.

\subsection{Robust Active Feature Acquisition}
Above, we introduce our proposed action space grouping technique and a partially observed OOD detection algorithm. Combining those components, we can now actively acquire features for problem with a large action space and simultaneously detect OOD inputs using the acquired subset of features. However, the OOD detection performance might be suboptimal, since the agent is not informed of the detection goal and merely focuses on predicting the target variable $y$. The features that are informative for predicting the target might not be informative for distinguishing OOD inputs.

In order to guide the agent to acquire features that are informative for OOD detection, we propose to utilize the likelihood, $p(s_1,\ldots,s_L \mid \I_m)$, as an auxiliary reward, which encourages the agent to acquire features more closely resemble the in-distribution ones, and thus reduces the false positive detection. Alternatively, if reducing the false negative is desired, we could use the negative likelihood $-p(s_1, \ldots, s_L \mid \I_m)$ as the auxiliary reward.

\begin{wrapfigure}{r}{0.35\linewidth}
\centering
\vspace{-10pt}
\includegraphics[width=\linewidth]{imgs/robust_afa.png}
\vspace{-10pt}
\caption{Schematic illustration of our robust AFA framework.}
\label{fig:robust_afa}
\end{wrapfigure}

In summary, our robust AFA framework contains a dynamics model, an OOD detector and an RL agent. The dynamics model captures the arbitrary conditionals, $p(x_u, y \mid x_o)$, and is utilized to provide auxiliary information and intermediate rewards. It also enables a simple and efficient action space grouping technique and thus scales AFA up to applications with large action spaces. The partially observed OOD detector is used to distinguish OOD inputs alongside the acquisition procedure and also used to provide an auxiliary reward so that the agent is encouraged to acquire informative features for OOD detection. The RL agent takes in the current acquired features and auxiliary information from the dynamics model and predicts what next feature to acquire. When the feature is actually acquired, the agent pays the acquisition cost of the feature and receives an intermediate reward from the dynamics model. When the acquisition process is terminated, the agent makes a final prediction about the target, $y$, using all its acquired features and receives an reward about its prediction. It also receives an reward from the OOD detector about the likelihood of the acquired feature subset in the transformed space (i.e., the norm of the scores). Please refer to Algorithm~\ref{alg:robust_afa} for additional details and to Fig.~\ref{fig:robust_afa} for an illustration.

In GSMRL \cite{li2020active}, the acquisition procedure is terminated when the agent selects a special termination action, which means each instance could have different number of features acquired. Although intriguing for practical use, it introduces additional complexity to assess OOD detection performance, since we need to separate two possible causes of a detection failure, i.e., not sufficient acquisitions and not effective acquisitions. To simplify the evaluation, we instead specify a fixed acquisition budget (i.e., the number of acquired features). The agent will terminate the acquisition process when it exceeds the specified acquisition budget. However, it is possible to incorporate a termination action into our framework.

\section{Experiments}
In this section, we evaluate our framework on several commonly used OOD detection benchmarks. Our model actively acquires features to predict the target and meanwhile determines whether the input is OOD using only the acquired features. Given that these benchmarks typically have a large number of candidate features, current AFA approaches cannot be applied directly. We instead compare to a modified GSMRL algorithm, where candidate features are clustered with our proposed action space grouping technique. We also compare to a simple random acquisition baseline, where a random unobserved feature is acquired at each acquisition step. The random policy is repeated for 5 times and the metrics are averaged from different runs. Please refer to Appendix~\ref{sec:appendix_exp} for experimental details. For each dataset, we assess the performance under several prespecified acquisition budgets. For classification task, the performance is evaluated by the classification accuracy; for reconstruction task, the performance is evaluated by the reconstruction MSE. We also detect OOD inputs using the acquired features and report the AUROC scores.

\begin{figure}
    \centering
    \begin{minipage}{\linewidth}
    \subfigure[MNIST]{\includegraphics[width=0.3\linewidth]{imgs/mnist_acc.png}}
    \hspace{3pt}
    \subfigure[FMNIST]{\includegraphics[width=0.3\linewidth]{imgs/fmnist_acc.png}}
    \hspace{3pt}
    \subfigure[SVHN]{\includegraphics[width=0.3\linewidth]{imgs/svhn_acc.png}}
    \vspace{-5pt}
    \caption{Classification accuracy for acquiring different number of features.}
    \label{fig:acc}
    \end{minipage}
    \begin{minipage}{\linewidth}
    \subfigure[MNIST - Omniglot]{\includegraphics[width=0.3\linewidth]{imgs/mnist_auc.png}}
    \hspace{3pt}
    \subfigure[FMNIST - MNIST]{\includegraphics[width=0.3\linewidth]{imgs/fmnist_auc.png}}
    \hspace{3pt}
    \subfigure[SVHN - CIFAR10]{\includegraphics[width=0.3\linewidth]{imgs/svhn_auc.png}}
    \vspace{-5pt}
    \caption{AUROC for OOD detection with acquired features.}
    \label{fig:auc}
    \end{minipage}
    \vspace{-12pt}
\end{figure}

\begin{wrapfigure}{r}{0.5\linewidth}
    \centering
    \vspace{-15pt}
    \includegraphics[width=\linewidth]{imgs/mnist_rafa.png}
    
    \vspace{5pt}
    
    \includegraphics[width=\linewidth]{imgs/fmnist_rafa.png}
    
    \vspace{5pt}
    
    \includegraphics[width=\linewidth]{imgs/svhn_rafa.png}
    \vspace{-10pt}
    \caption{Examples of the acquisition process from our robust AFA framework. The bar charts demonstrate the prediction probability at the corresponding acquisition step.}
    \label{fig:rafa}
\end{wrapfigure}

\paragraph{Robust Active Feature Acquisition}
We first evaluate the AFA tasks using several classification datasets. The agent is trained to acquire the pixel values. For color images, the agent acquires all three channels at once. For MNIST \cite{lecun2010mnist} and FMNIST \cite{xiao2017/online}, we follow GSMRL to train the surrogate model using a class conditioned ACFlow \cite{li2020acflow}; for SVHN \cite{Netzer2011}, we simply use a partially observed classifier to learn $p(y \mid x_o)$ since we found ACFlow difficult to train for this dataset. The auxiliary information is accordingly modified to contain only the prediction probability. Figure~\ref{fig:acc} and \ref{fig:auc} report the classification accuracy and OOD detection AUROC respectively. The accuracy is significantly higher for RL approaches than the random acquisition policy. Although we expect a trade-off between accuracy and OOD detection performance for our robust AFA framework, the accuracy is actually comparable to GSMRL and sometimes even better across the datasets. Meanwhile, the OOD detection performance for our robust AFA framework is significantly improved by enforcing the agent to acquire informative features for OOD identification. For SVHN and CIFAR10 detection, the AUROC for GSMRL is even lower than the random policy, which we believe is because of the discrepancy of informative features for two different goals. Augmented with the detector reward solves the problem and improves the detection performance even further.
Figure~\ref{fig:rafa} presents several examples of the acquisition process from our robust AFA framework. We can see the prediction becomes certain after only a few acquisition steps. See appendix~\ref{sec:appendix_exp} for additional examples.


\begin{figure}
\begin{minipage}{0.49\linewidth}
    \centering
    \subfigure[MNIST]{\includegraphics[width=0.49\linewidth]{imgs/mnist_mse.png}}
    \subfigure[FMNIST]{\includegraphics[width=0.49\linewidth]{imgs/fmnist_mse.png}}
    \vspace{-5pt}
    \caption{Reconstruction MSE for robust AIR.}
    \label{fig:mse}
\end{minipage}
\begin{minipage}{0.49\linewidth}
    \centering
    \subfigure[MNIST - Omniglot]{\includegraphics[width=0.49\linewidth]{imgs/mnist_rec_auc.png}}
    \subfigure[FMNIST - MNIST]{\includegraphics[width=0.49\linewidth]{imgs/fmnist_rec_auc.png}}
    \vspace{-7pt}
    \caption{OOD detection for robust AIR.}
    \label{fig:rec}
\end{minipage}
\vspace{-10pt}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=0.49\linewidth]{imgs/mnist_rair.png}
    \includegraphics[width=0.49\linewidth]{imgs/fmnist_rair.png}
    \vspace{-5pt}
    \caption{Examples of the acquisition process for robust AIR.}
    \label{fig:rair}
    \vspace{-8pt}
\end{figure}

\paragraph{Robust Active Instance Recognition}
In this section, we evaluate the AIR task using MNIST and FashionMNSIT datasets. Following GSMRL \cite{li2020active}, we use ACFlow as the surrogate model. Figure~\ref{fig:mse} and \ref{fig:rec} report the reconstruction MSE and OOD detection performance respectively using the acquired features. We can see our robust AIR framework improves the OOD detection performance significantly, especially when the acquisition budget is low, while the reconstruction MSEs are comparable to GSMRL. Figure~\ref{fig:rair} presents several examples of the acquisition process for robust AIR.


\begin{wrapfigure}{r}{0.3\linewidth}
    \centering
    \vspace{-10pt}
    \includegraphics[width=\linewidth]{imgs/ablation_acc.png}
    \vspace{-13pt}
    \caption{Compare AFA performance with or without action grouping.}
    \label{fig:ablation_acc}
\end{wrapfigure}

\paragraph{Ablations}
% \paragraph{Action Space Grouping}
Our proposed action grouping technique enables the agent to acquire features from a potentially large pool of candidates. However, it also introduces some complexity due to the autoregressive factorization in \eqref{eq:autoreg_action}. In Fig.~\ref{fig:ablation_acc}, we compare two agents with and without the action grouping using a downsampled MNIST. We can see the action grouping does not degrade the performance on smaller dimensionalities whilst allowing one to work over larger dimensionalities that previous methods cannot scale to.

% \paragraph{Partially Observed OOD Detection}
Although our PO-MSMA is designed for partially observed instances, it can handle fully observed ones as special cases. In Table~\ref{tab:ablation_pomsma}, we report the AUROC scores for both methods. We can see our PO-MSMA is competitive even though it is not trained to detect fully observed instances.

\begin{table}[H]
    \centering
    \vspace{-8pt}
    \caption{Comparison with MSMA for fully observed OOD detection. AUROC scores are reported.}
    \label{tab:ablation_pomsma}
    \small
    \begin{tabular}{c|c|c|c|c}
    \toprule
         &  MNIST - Omniglot & FMNIST - MNIST & SVHN - CIFAR10 & CIFAR10 - SVHN\\
    \midrule
        MSMA & - & 82.56 & 97.60 & 95.50 \\
        PO-MSMA & 99.55 & 96.62 & 97.77 & 74.74 \\
    \bottomrule
    \end{tabular}
\end{table}

\section{Discussion and Conclusion}\label{sec:conclusion}
In this work, we investigate an understudied problem in AFA, where we  switch gears from acquiring informative features to acquiring robust features. A robust AFA framework is proposed here to acquire feature actively and determine whether the input is OOD using only the acquired subset of features. In order to scale up the AFA models to practical use, we develop a hierarchical acquisition policy, where the candidate features are grouped together based on their relevance to the target. Our framework represents the first AFA model that can deal with a potentially large pool of candidate features. Extensive experiments are conducted to showcase the effectiveness of our framework. Due to the fact that instances are partially observed for AFA, our framework is not guaranteed to be robust for adversaries (a particular type of OOD with minimum modification to a valid input), since the adversary can easily modify those unobserved features to fool the detector. However, the existence of an adversarially robust AFA model is also an open question.

\begin{ack}
N/A
\end{ack}

\bibliographystyle{unsrt}
\bibliography{neurips_2021}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Checklist}

\begin{enumerate}

\item For all authors...
\begin{enumerate}
  \item Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
    \answerYes{}
  \item Did you describe the limitations of your work?
    \answerYes{See Sec.~\ref{sec:conclusion}}
  \item Did you discuss any potential negative societal impacts of your work?
    \answerYes{See appendix \ref{sec:impact}}
  \item Have you read the ethics review guidelines and ensured that your paper conforms to them?
    \answerYes{}
\end{enumerate}

\item If you are including theoretical results...
\begin{enumerate}
  \item Did you state the full set of assumptions of all theoretical results?
    \answerNA{}
	\item Did you include complete proofs of all theoretical results?
    \answerNA{}
\end{enumerate}

\item If you ran experiments...
\begin{enumerate}
  \item Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)?
    \answerYes{Code is provided as supplemental material.}
  \item Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?
    \answerYes{See appendix \ref{sec:appendix_exp}}
	\item Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)?
    \answerYes{See appendix \ref{sec:appendix_exp}}
	\item Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?
    \answerYes{See appendix \ref{sec:appendix_exp}}
\end{enumerate}

\item If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
\begin{enumerate}
  \item If your work uses existing assets, did you cite the creators?
    \answerYes{All datasets used in this work are publically available.}
  \item Did you mention the license of the assets?
    \answerNA{}
  \item Did you include any new assets either in the supplemental material or as a URL?
    \answerNA{}
  \item Did you discuss whether and how consent was obtained from people whose data you're using/curating?
    \answerNA{}
  \item Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content?
    \answerNA{}
\end{enumerate}

\item If you used crowdsourcing or conducted research with human subjects...
\begin{enumerate}
  \item Did you include the full text of instructions given to participants and screenshots, if applicable?
    \answerNA{}
  \item Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable?
    \answerNA{}
  \item Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation?
    \answerNA{}
\end{enumerate}

\end{enumerate}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\clearpage
\appendix

\section{Broader Impact}\label{sec:impact}

\section{Experimental Details}\label{sec:appendix_exp}


\end{document}
