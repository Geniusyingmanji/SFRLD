clc
clear
x = 1:11;

A=[76.77	75.84	77.08	77.01
77.39	77.08	76.85	76.92
76.54	76.46	76.46	76.62
75.69	75.69	75.92	75.77
75.54	75.54	75.61	75.46
75.23	75.23	75.23	75.23
75.23	75.23	75.23	75.23
75.07	75.15	75.15	75.07
75.23	75.15	75.07	75.07
74.76	74.76	74.76	74.76
74.76	74.76	74.76	74.76
];
A1=[
63.75
76.25
76.25
76.25
76.25
];
A2=[
60.62
60.62
60.62
60.62
60.62
];
A3=[
54.37
56.25
56.25
56.25
56.25
];
A4=[
45.62
99.38
99.38
99.38
99.38
];
A5=[
79.37
79.37
79.37
79.37
79.37
];
A6=[
67.5
74.38
74.38
74.38
74.38
];
A7=[
58.75
84.38
84.38
84.38
84.38
];
A8=[
67.5
86.88
86.88
86.88
86.88
];
A9=[
45
85
85
85
85
];
x=1:5
plot(x, A1(:,1), '-r', x, A2(:,1), '--r', x, A3(:,1), '-r*', x, A4(:,1), '-g',x, A5(:,1), '--g',x, A6(:,1), '-g*',x, A7(:,1), '-b',x, A8(:,1), '--b',x, A9(:,1), '-b*', 'LineWidth', 2);
legend('subject1', 'subject2', 'subject3', 'subject4','subject5','subject6','subject7','subject8','subject9');
% set(gca,'xticklabel',[1:5]);
xlabel('#Iteration','FontSize',14);
ylabel('Classifcation accuracy (%)','FontSize',14)
grid on
axis([1 5 40 105]);
