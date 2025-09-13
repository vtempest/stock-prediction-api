import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/',
    component: ComponentCreator('/', 'b76'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', '9e7'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', 'e4e'),
            routes: [
              {
                path: '/functions/',
                component: ComponentCreator('/functions/', '736'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/functions/api-client/dist',
                component: ComponentCreator('/functions/api-client/dist', 'e6c'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/functions/call-predict',
                component: ComponentCreator('/functions/call-predict', 'd02'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/functions/export-db-daily-total',
                component: ComponentCreator('/functions/export-db-daily-total', '94d'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/functions/export-db-status',
                component: ComponentCreator('/functions/export-db-status', '149'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/functions/modules',
                component: ComponentCreator('/functions/modules', '563'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/research/Feature Engineering',
                component: ComponentCreator('/research/Feature Engineering', '65e'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/research/Model Comparison',
                component: ComponentCreator('/research/Model Comparison', '3b5'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/research/Variables Biogas Output',
                component: ComponentCreator('/research/Variables Biogas Output', 'b46'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/research/Weighted_Ensemble_Improvements',
                component: ComponentCreator('/research/Weighted_Ensemble_Improvements', 'f36'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/research/XGBoost Params',
                component: ComponentCreator('/research/XGBoost Params', 'd76'),
                exact: true,
                sidebar: "default"
              },
              {
                path: '/',
                component: ComponentCreator('/', '48d'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
