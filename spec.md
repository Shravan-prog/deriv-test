# customer support ticket urgency

## goal
buld a python pipeline that classifies customer support tickets, routes uncertain tickets to human review , generate replies for safe ticket, saves results and compute evaluation metrix

## input
ticket.json
 - ticket_id
 - message
 - expected category
 - expected urgency level

 output per ticket:
  - ticket_id
  - category
  - urgency
  - confidence
  - route
  - suggested reply
  - error if applicable

  So the pipeline stages will be like inputs will be loaded and text will be pre-processed, and we'll have our model prompts and structured output will be passed, and we'll have our confidence, we'll have our confidence scores, and after that we'll have our routed and response generated, and we'll have our results saved and evaluation completed, and we'll have our validation completed. So then after that we'll have routing. So in routing, if confidence is greater than confidence threshold, then route will be auto, or otherwise route should be human review. So suggested reply will be null. Reproducibility: we'll have deterministic text pre-processing and stable ticket ordering, explicit configuration, store raw structured model responses, and we'll support deterministic mock or fallback mode, and we'll have no dependence on sample working wording. So for metrics we can have category accuracy, we can have urgency accuracy, we'll have coverage, and we'll have human review rate. So what will be the definition of done here? So running this pipeline, so running the pipeline from a clean structured output, routing decision, stage transmission, saved results, and evaluation metrics.