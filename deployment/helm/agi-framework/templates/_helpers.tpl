{{/* =============================================================================
AGI Unified Framework - Helm Chart Helpers
=============================================================================
通用的Helm模板辅助函数，用于生成标准标签、名称、选择器等
============================================================================= */}}

{{/* =============================================================================
基础名称生成
============================================================================= */}}

{{/*
生成Chart的基础名称
如果 .Values.nameOverride 存在则使用，否则使用Chart名称
*/}}
{{- define "agi-framework.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
生成完整的资源名称
包含Release名称和Chart名称（如果未覆盖）
*/}}
{{- define "agi-framework.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
生成Chart名称和版本
用于Helm的chart标签
*/}}
{{- define "agi-framework.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* =============================================================================
标签生成
============================================================================= */}}

{{/*
生成标准Kubernetes标签
包含应用标签、版本标签、管理标签等
*/}}
{{- define "agi-framework.labels" -}}
helm.sh/chart: {{ include "agi-framework.chart" . }}
{{ include "agi-framework.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: agi-unified-framework
{{- if .Values.global.commonLabels }}
{{ toYaml .Values.global.commonLabels }}
{{- end }}
{{- end }}

{{/*
生成选择器标签
用于Deployment、Service等资源的选择器
*/}}
{{- define "agi-framework.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agi-framework.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API服务特定标签
*/}}
{{- define "agi-framework.api.labels" -}}
{{ include "agi-framework.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Worker服务特定标签
*/}}
{{- define "agi-framework.worker.labels" -}}
{{ include "agi-framework.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Beat服务特定标签
*/}}
{{- define "agi-framework.beat.labels" -}}
{{ include "agi-framework.labels" . }}
app.kubernetes.io/component: scheduler
{{- end }}

{{/* =============================================================================
名称生成辅助函数
============================================================================= */}}

{{/*
API服务名称
*/}}
{{- define "agi-framework.api.name" -}}
{{- printf "%s-api" (include "agi-framework.fullname" .) }}
{{- end }}

{{/*
Worker服务名称
*/}}
{{- define "agi-framework.worker.name" -}}
{{- printf "%s-worker" (include "agi-framework.fullname" .) }}
{{- end }}

{{/*
Beat服务名称
*/}}
{{- define "agi-framework.beat.name" -}}
{{- printf "%s-beat" (include "agi-framework.fullname" .) }}
{{- end }}

{{/*
ConfigMap名称
*/}}
{{- define "agi-framework.configmap.name" -}}
{{- printf "%s-config" (include "agi-framework.fullname" .) }}
{{- end }}

{{/*
Secret名称
*/}}
{{- define "agi-framework.secret.name" -}}
{{- printf "%s-secrets" (include "agi-framework.fullname" .) }}
{{- end }}

{{/*
服务账户名称
*/}}
{{- define "agi-framework.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agi-framework.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/* =============================================================================
镜像生成
============================================================================= */}}

{{/*
生成完整的镜像地址
包含仓库、镜像名和标签
*/}}
{{- define "agi-framework.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .repository }}
{{- $tag := .tag | default "latest" }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/* =============================================================================
环境变量生成
============================================================================= */}}

{{/*
生成标准环境变量
*/}}
{{- define "agi-framework.env" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- end }}

{{/* =============================================================================
资源请求和限制生成
============================================================================= */}}

{{/*
生成资源请求和限制
*/}}
{{- define "agi-framework.resources" -}}
{{- if .resources }}
resources:
  {{- toYaml .resources | nindent 2 }}
{{- end }}
{{- end }}

{{/* =============================================================================
探针生成
============================================================================= */}}

{{/*
生成存活探针
*/}}
{{- define "agi-framework.livenessProbe" -}}
{{- if .livenessProbe.enabled }}
livenessProbe:
  httpGet:
    path: {{ .livenessProbe.path }}
    port: http
    httpHeaders:
      - name: Accept
        value: application/json
  initialDelaySeconds: {{ .livenessProbe.initialDelaySeconds }}
  periodSeconds: {{ .livenessProbe.periodSeconds }}
  timeoutSeconds: {{ .livenessProbe.timeoutSeconds }}
  failureThreshold: {{ .livenessProbe.failureThreshold }}
  successThreshold: {{ .livenessProbe.successThreshold | default 1 }}
{{- end }}
{{- end }}

{{/*
生成就绪探针
*/}}
{{- define "agi-framework.readinessProbe" -}}
{{- if .readinessProbe.enabled }}
readinessProbe:
  httpGet:
    path: {{ .readinessProbe.path }}
    port: http
    httpHeaders:
      - name: Accept
        value: application/json
  initialDelaySeconds: {{ .readinessProbe.initialDelaySeconds }}
  periodSeconds: {{ .readinessProbe.periodSeconds }}
  timeoutSeconds: {{ .readinessProbe.timeoutSeconds }}
  failureThreshold: {{ .readinessProbe.failureThreshold }}
  successThreshold: {{ .readinessProbe.successThreshold | default 1 }}
{{- end }}
{{- end }}

{{/* =============================================================================
安全上下文生成
============================================================================= */}}

{{/*
生成Pod安全上下文
*/}}
{{- define "agi-framework.podSecurityContext" -}}
{{- if .podSecurityContext }}
securityContext:
  {{- toYaml .podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
生成容器安全上下文
*/}}
{{- define "agi-framework.securityContext" -}}
{{- if .securityContext }}
securityContext:
  {{- toYaml .securityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/* =============================================================================
亲和性生成
============================================================================= */}}

{{/*
生成亲和性配置
*/}}
{{- define "agi-framework.affinity" -}}
{{- if .affinity }}
affinity:
  {{- toYaml .affinity | nindent 2 }}
{{- end }}
{{- end }}

{{/*
生成容忍配置
*/}}
{{- define "agi-framework.tolerations" -}}
{{- if .tolerations }}
tolerations:
  {{- toYaml .tolerations | nindent 2 }}
{{- end }}
{{- end }}

{{/*
生成拓扑分布约束
*/}}
{{- define "agi-framework.topologySpreadConstraints" -}}
{{- if .topologySpreadConstraints }}
topologySpreadConstraints:
  {{- toYaml .topologySpreadConstraints | nindent 2 }}
{{- end }}
{{- end }}

{{/* =============================================================================
存储生成
============================================================================= */}}

{{/*
生成持久化卷声明名称
*/}}
{{- define "agi-framework.pvc.name" -}}
{{- printf "%s-data" (include "agi-framework.fullname" .) }}
{{- end }}

{{/* =============================================================================
Ingress生成
============================================================================= */}}

{{/*
生成Ingress的TLS配置
*/}}
{{- define "agi-framework.ingress.tls" -}}
{{- if .tls }}
tls:
  {{- range .tls }}
  - hosts:
      {{- range .hosts }}
      - {{ . | quote }}
      {{- end }}
    secretName: {{ .secretName }}
  {{- end }}
{{- end }}
{{- end }}

{{/* =============================================================================
验证和检查
============================================================================= */}}

{{/*
验证必需值
*/}}
{{- define "agi-framework.validate" -}}
{{- if not .Values.api.image.repository }}
  {{- fail "api.image.repository is required" }}
{{- end }}
{{- if not .Values.api.image.tag }}
  {{- fail "api.image.tag is required" }}
{{- end }}
{{- end }}
