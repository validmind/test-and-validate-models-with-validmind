{{/*
Common name prefix for all resources in this release.
*/}}
{{- define "quickstart.name" -}}
{{- .Release.Name -}}
{{- end -}}

{{/*
Workbench (Notebook CR) name — also the StatefulSet/pod prefix the
notebook controller derives from it.
*/}}
{{- define "quickstart.workbenchName" -}}
{{- printf "%s-workbench" .Release.Name -}}
{{- end -}}

{{/*
Name of the connection-shaped Secret carrying the ValidMind credentials.
*/}}
{{- define "quickstart.connectionName" -}}
{{- printf "%s-validmind-connection" .Release.Name -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "quickstart.labels" -}}
app.kubernetes.io/name: {{ include "quickstart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}
