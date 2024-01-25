from sentence_transformers import SentenceTransformer
from angle_emb import AnglE
from transformers import AutoTokenizer
import csv
import numpy as np
import torch

random_words = [
    'Ability',
    'Absence',
    'Abundance',
    'Acceptance',
    'Accomplishment',
    'Accuracy',
    'Achievement',
    'Admiration',
    'Adoration',
    'Adventure',
    'Affection',
    'Aggression',
    'Agony',
    'Altruism',
    'Ambition',
    'Amusement',
    'Anguish',
    'Annoyance',
    'Anxiety',
    'Apprehension',
    'Approval',
    'Awe',
    'Beauty',
    'Belief',
    'Betrayal',
    'Bliss',
    'Boldness',
    'Boredom',
    'Bravery',
    'Brilliance',
    'Calmness',
    'Candor',
    'Capability',
    'Care',
    'Caution',
    'Celebration',
    'Challenge',
    'Change',
    'Chaos',
    'Charisma',
    'Charm',
    'Chastity',
    'Cheerfulness',
    'Clarity',
    'Closure',
    'Comfort',
    'Commitment',
    'Compassion',
    'Complacency',
    'Compromise',
    'Concentration',
    'Confidence',
    'Confusion',
    'Conscience',
    'Consolation',
    'Contentment',
    'Contradiction',
    'Control',
    'Conviction',
    'Cooperation',
    'Courage',
    'Courtesy',
    'Curiosity',
    'Danger',
    'Darkness',
    'Deceit',
    'Dedication',
    'Defeat',
    'Defiance',
    'Delight',
    'Democracy',
    'Denial',
    'Dependence',
    'Depression',
    'Desire',
    'Despair',
    'Determination',
    'Devotion',
    'Dignity',
    'Diligence',
    'Disappointment',
    'Disbelief',
    'Discord',
    'Discretion',
    'Discrimination',
    'Disgust',
    'Dishonesty',
    'Disillusionment',
    'Dislike',
    'Disobedience',
    'Disrespect',
    'Dissatisfaction',
    'Diversity',
    'Divinity',
    'Dream',
    'Drive',
    'Duty',
    'Ecstasy',
    'Education',
    'Elegance',
    'Empathy',
    'Encouragement',
    'Endurance',
    'Energy',
    'Enthusiasm',
    'Envy',
    'Equality',
    'Escape',
    'Ethics',
    'Euphoria',
    'Evil',
    'Evolution',
    'Excellence',
    'Excitement',
    'Exhilaration',
    'Expectation',
    'Experience',
    'Expression',
    'Extravagance',
    'Failure',
    'Faith',
    'Fame',
    'Fascination',
    'Fashion',
    'Fatigue',
    'Fear',
    'Ferocity',
    'Fidelity',
    'Fierceness',
    'Folly',
    'Foolishness',
    'Forgiveness',
    'Formality',
    'Fortune',
    'Freedom',
    'Friendship',
    'Frustration',
    'Fun',
    'Fury',
    'Generosity',
    'Gentleness',
    'Glory',
    'Goodness',
    'Grace',
    'Gratitude',
    'Greed',
    'Grief',
    'Growth',
    'Guilt',
    'Happiness',
    'Harmony',
    'Hatred',
    'Healing',
    'Health',
    'Heartbreak',
    ' Idealism',
    'Happiness',
    'Hardship',
    'Harmony',
    'Health',
    'Heartache',
    'Heat',
    'Heroism',
    'Honesty',
    'Honor',
    'Hope',
    'Hopelessness',
    'Hospitality',
    'Humility',
    'Humor',
    'Hurt',
    'Identity',
    'Ignorance',
    'Illusion',
    'Imagination',
    'Impatience',
    'Independence',
    'Indifference',
    'Individuality',
    'Infinity',
    'Injustice',
    'Innocence',
    'Innovation',
    'Insight',
    'Inspiration',
    'Instinct',
    'Integrity',
    'Intensity',
    'Intention',
    'Intuition',
    'Invention',
    'Irresponsibility',
    'Isolation',
    'Jealousy',
    'Joy',
    'Judgment',
    'Justice',
    'Kindness',
    'Knowledge',
    'Laughter',
    'Leadership',
    'Learning',
    'Liberation',
    'Liberty',
    'Life',
    'Light',
    'Loneliness',
    'Longing',
    'Love',
    'Loyalty',
    'Madness',
    'Magic',
    'Magnificence',
    'Majesty',
    'Malice',
    'Manners',
    'Martyrdom',
    'Masculinity',
    'Materialism',
    'Maturity',
    'Meaning',
    'Mediocrity',
    'Melancholy',
    'Menace',
    'Mercy',
    'Merit',
    'Meticulousness',
    'Mindfulness',
    'Mischief',
    'Misery',
    'Modesty',
    'Momentum',
    'Morality',
    'Motivation',
    'Mystery',
    'Nastiness',
    'Nature',
    'Necessity',
    'Negativity',
    'Nerve',
    'Nostalgia',
    'Novelty',
    'Nurture',
    'Obscurity',
    'Obsession',
    'Obedience',
    'Oblivion',
    'Obscurity',
    'Openness',
    'Optimism',
    'Order',
    'Originality',
    'Outrage',
    'Pain',
    'Panic',
    'Passion',
    'Patience',
    'Patriotism',
    'Peace',
    'Perfection',
    'Perseverance',
    'Persistence',
    'Persuasion',
    'Pessimism',
    'Philosophy',
    'Piety',
    'Playfulness',
    'Pleasure',
    'Poise',
    'Politeness',
    'Popularity',
    'Positivity',
    'Power',
    'Practicality',
    'Pragmatism',
    'Precision',
    'Prejudice',
    'Preparation',
    'Presence',
    'Pride',
    'Privacy',
    'Progress',
    'Prosperity',
    'Protection',
    'Purity',
    'Quality',
    'Quantity',
    'Quietude',
    'Quirks',
    'Rage',
    'Rationality',
    'Realism',
    'Reason',
    'Rebellion',
    'Reconciliation',
    'Refinement',
    'Reflection',
    'Regret',
    'Rejection',
    'Relief',
    'Religion',
    'Renunciation',
    'Reputation',
    'Resentment',
    'Resilience',
    'Resolution',
    'Respect',
    'Responsibility',
    'Rest',
    'Restraint',
    'Revelation',
    'Reverence',
    'Revolution',
    'Richness',
    'Rigor',
    'Risk',
    'Romance',
    'Royalty',
    'Sacrifice',
    'Sadness',
    'Safety',
    'Sanity',
    'Satisfaction',
    'Science',
    'Security',
    'Sacrifice',
    'Sadness',
    'Safety',
    'Sanity',
    'Satisfaction',
    'Science',
    'Security',
    'Seduction',
    'Self-control',
    'Self-esteem',
    'Selflessness',
    'Sensibility',
    'Sensitivity',
    'Serenity',
    'Service',
    'Sexuality',
    'Shame',
    'Silence',
    'Simplicity',
    'Sincerity',
    'Skill',
    'Sobriety',
    'Solitude',
    'Sophistication',
    'Soul',
    'Sparkle',
    'Speed',
    'Spirit',
    'Spontaneity',
    'Stability',
    'Stamina',
    'Status',
    'Strength',
    'Struggle',
    'Style',
    'Subtlety',
    'Success',
    'Suffering',
    'Support',
    'Surprise',
    'Surrender',
    'Survival',
    'Sustainability',
    'Sympathy',
    'System',
    'Talent',
    'Taste',
    'Teaching',
    'Technology',
    'Temptation',
    'Tenderness',
    'Terror',
    'Testimony',
    'Thirst',
    'Thoughtfulness',
    'Thrill',
    'Time',
    'Tolerance',
    'Tradition',
    'Tranquility',
    'Transcendence',
    'Transformation',
    'Transparency',
    'Trust',
    'Truth',
    'Understanding',
    'Uniqueness',
    'Unity',
    'Universalism',
    'Unpredictability',
    'Unselfishness',
    'Urgency',
    'Usefulness',
    'Utility',
    'Valor',
    'Vanity',
    'Variety',
    'Vegetation',
    'Vengeance',
    'Veracity',
    'Versatility',
    'Vibrancy',
    'Victory',
    'Vigilance',
    'Vim',
    'Virtue',
    'Vision',
    'Vitality',
    'Vivacity',
    'Voice',
    'Vulnerability',
    'Warmth',
    'Wealth',
    'Will',
    'Willpower',
    'Wisdom',
    'Wonder',
    'Worry',
    'Worship',
    'Worthiness',
    'Xenophobia',
    'Yearning',
    'Yield',
    'Youthfulness',
    'Zeal',
    'Zenith',
    'Zest',
    'Entrepreneurship',
    'Innovation',
    'Creativity',
    'Leadership',
    'Management',
    'Marketing',
    'Sales',
    'Customer service',
    'Advertising',
    'Public relations',
    'Branding',
    'Strategy',
    'Operations',
    'Finance',
    'Accounting',
    'Investing',
    'Banking',
    'Insurance',
    'Real estate',
    'Law',
    'Criminal law',
    'Civil law',
    'Constitutional law',
    'International law',
    'Humanitarian law',
    'Intellectual property law',
    'Corporate law',
    'Tax law',
    'Immigration law',
    'Family law',
    'Estate planning',
    'Employment law',
    'Education',
    'Early childhood education',
    'Elementary education',
    'Secondary education',
    'Higher education',
    'Adult education',
    'Vocational education',
    'Special education',
    'Curriculum',
    'Instruction',
    'Assessment',
    'Standardized testing',
    'Learning styles',
    'Teaching methods',
    'Classroom management',
    'Educational technology',
    'Distance learning',
    'Blended learning',
    'Library science',
    'Information science',
    'Data science',
    'Computer science',
    'Artificial intelligence',
    'Machine learning',
    'Robotics',
    'Cybersecurity',
    'Web development',
    'Software development',
    'Mobile app development',
    'User experience',
    'User interface',
    'Graphic design',
    'Video game design',
    'Multimedia',
    'Animation',
    'Visual effects',
    'Music production',
    'Sound design',
    'Film production',
    'Screenwriting',
    'Acting',
    'Directing',
    'Cinematography',
    'Editing',
    'Special effects',
    'Costume design',
    'Makeup artistry',
    'Set design.',
    'Altruism',
    'Empathy',
    'Compassion',
    'Kindness',
    'Generosity',
    'Philanthropy',
    'Charity',
    'Volunteerism',
    'Service',
    'Gratitude',
    'Forgiveness',
    'Humility',
    'Modesty',
    'Respect',
    'Dignity',
    'Honor',
    'Integrity',
    'Authenticity',
    'Sincerity',
    'Transparency',
    'Accountability',
    'Responsibility',
    'Stewardship',
    'Environmentalism',
    'Sustainability',
    'Conservation',
    'Biodiversity',
    'Ecology',
    'Ecosystem',
    'Climate change',
    'Global warming',
    'Pollution',
    'Waste',
    'Recycling',
    'Renewable energy',
    'Non-renewable energy',
    'Fossil fuels',
    'Nuclear energy',
    'Hydroelectric power',
    'Wind power',
    'Solar power',
    'Geothermal energy',
    'Biomass energy',
    'Energy efficiency',
    'Green buildings',
    'Transportation',
    'Electric vehicles',
    'Public transportation',
    'Bicycling',
    'Walking',
    'Health',
    'Wellness',
    'Fitness',
    'Nutrition',
    'Sleep',
    'Stress management',
    'Mental health',
    'Therapy',
    'Counseling',
    'Medication',
    'Addiction recovery',
    'Rehabilitation',
    'Disability',
    'Accessibility',
    'Inclusivity',
    'Equality',
    'Equity',
    'Diversity',
    'Human rights',
    'Civil rights',
    'Social justice',
    'Racial justice',
    'Gender equality',
    'LGBTQ+ rights',
    'Women’s rights',
    'Children’s rights',
    'Animal rights',
    'Environmental justice',
    'Peace',
    'Non-violence',
    'Conflict resolution',
    'Mediation',
    'Negotiation',
    'Diplomacy',
    'War',
    'Violence',
    'Aggression',
    'Bullying',
    'Harassment',
    'Discrimination',
    'Prejudice',
    'Stereotyping',
    'Racism',
    'Sexism',
    'Homophobia',
    'Transphobia',
    'Ableism',
    'Ageism',
    'Classism',
    'Xenophobia.',
    'Immateriality',
    'Abstraction',
    'Conceptuality',
    'Ideality',
    'Metaphysics',
    'Ontology',
    'Epistemology',
    'Logic',
    'Semiotics',
    'Linguistics',
    'Phonology',
    'Morphology',
    'Syntax',
    'Semantics',
    'Pragmatics',
    'Discourse',
    'Narrative',
    'Genre',
    'Style',
    'Form',
    'Structure',
    'Content',
    'Meaning',
    'Significance',
    'Context',
    'Interpretation',
    'Hermeneutics',
    'Exegesis',
    'Critique',
    'Analysis',
    'Synthesis',
    'Evaluation',
    'Judgment',
    'Aesthetics',
    'Beauty',
    'Harmony',
    'Proportion',
    'Symmetry',
    'Unity',
    'Diversity',
    'Originality',
    'Innovation',
    'Tradition',
    'Convention',
    'Genre',
    'Form',
    'Medium',
    'Art',
    'Music',
    'Literature',
    'Poetry',
    'Drama',
    'Fiction',
    'Non-fiction',
    'Biography',
    'Autobiography',
    'Memoir',
    'Journalism',
    'Criticism',
    'Film',
    'Television',
    'Theatre',
    'Dance',
    'Painting',
    'Sculpture',
    'Photography',
    'Architecture',
    'Design',
    'Fashion',
    'Technology',
    'Science',
    'Biology',
    'Chemistry',
    'Physics',
    'Astronomy',
    'Psychology',
    'Sociology',
    'Anthropology',
    'History',
    'Geography',
    'Politics',
    'Government',
    'Democracy',
    'Dictatorship',
    'Monarchy',
    'Republic',
    'Authoritarianism',
    'Totalitarianism',
    'Fascism',
    'Communism',
    'Capitalism',
    'Socialism',
    'Anarchism',
    'Liberalism',
    'Conservatism',
    'Nationalism',
    'Patriotism',
    'Globalization',
    'Internationalism',
    'Humanism.',
    'Redemption',
    'Salvation',
    'Grace',
    'Mercy',
    'Justice',
    'Fairness',
    'Equality',
    'Equity',
    'Diversity',
    'Inclusion',
    'Tolerance',
    'Acceptance',
    'Openness',
    'Curiosity',
    'Creativity',
    'Imagination',
    'Innovation',
    'Inspiration',
    'Motivation',
    'Ambition',
    'Perseverance',
    'Dedication',
    'Commitment',
    'Sacrifice',
    'Endurance',
    'Patience',
    'Discipline',
    'Self-control',
    'Temperance',
    'Indulgence',
    'Hedonism',
    'Pleasure',
    'Satisfaction',
    'Indulgence',
    'Restraint',
    'Asceticism',
    'Self-denial',
    'Abstinence',
    'Sobriety',
    'Intoxication',
    'Addiction',
    'Dependence',
    'Freedom',
    'Liberty',
    'Autonomy',
    'Independence',
    'Sovereignty',
    'Authority',
    'Power',
    'Control',
    'Influence',
    'Domination',
    'Submission',
    'Obedience',
    'Compliance',
    'Rebellion',
    'Revolution',
    'Protest',
    'Activism',
    'Dissent',
    'Criticism',
    'Censorship',
    'Freedom of speech',
    'Privacy',
    'Security',
    'Surveillance',
    'Cynicism',
    'Skepticism',
    'Doubt',
    'Agnosticism',
    'Atheism',
    'Spirituality',
    'Religion',
    'Belief',
    'Faith',
    'Worship',
    'Prayer',
    'Ritual',
    'Meditation',
    'Enlightenment',
    'Wisdom',
    'Knowledge',
    'Understanding',
    'Insight',
    'Perception',
    'Intuition',
    'Logic',
    'Reason',
    'Argument',
    'Debate',
    'Persuasion',
    'Rhetoric',
    'Propaganda',
    'Manipulation',
    'Brainwashing',
    'Education',
    'Learning',
    'Intelligence',
    'Genius',
    'Brilliance.',
    'Nationality',
    'Ethnicity',
    'Culture',
    'Heritage',
    'Tradition',
    'Innovation',
    'Modernity',
    'Postmodernity',
    'Globalization',
    'Internationalism',
    'Nationalism',
    'Patriotism',
    'Identity',
    'Belonging',
    'Alienation',
    'Loneliness',
    'Solitude',
    'Isolation',
    'Community',
    'Society',
    'Family',
    'Friendship',
    'Love',
    'Romance',
    'Passion',
    'Affection',
    'Empathy',
    'Compassion',
    'Kindness',
    'Generosity',
    'Altruism',
    'Philanthropy',
    'Charity',
    'Gratitude',
    'Joy',
    'Happiness',
    'Contentment',
    'Satisfaction',
    'Fulfillment',
    'Enthusiasm',
    'Excitement',
    'Anticipation',
    'Fear',
    'Anxiety',
    'Dread',
    'Terror',
    'Panic',
    'Despair',
    'Hopelessness',
    'Sadness',
    'Grief',
    'Loss',
    'Sorrow',
    'Regret',
    'Guilt',
    'Shame',
    'Embarrassment',
    'Humiliation',
    'Vulnerability',
    'Resilience',
    'Courage',
    'Bravery',
    'Heroism',
    'Cowardice',
    'Betrayal',
    'Loyalty',
    'Trust',
    'Deceit',
    'Dishonesty',
    'Corruption',
    'Integrity',
    'Authenticity',
    'Sincerity',
    'Transparency',
    'Accountability',
    'Responsibility',
    'Obligation',
    'Duty',
    'Honor',
    'Dignity',
    'Respect',
    'Humility',
    'Modesty',
    'Ego',
    'Arrogance',
    'Narcissism',
    'Self-esteem',
    'Self-worth',
    'Confidence',
    'Doubt',
    'Insecurity',
    'Jealousy',
    'Envy',
    'Resentment',
    'Bitterness',
    'Forgiveness',
    'Reconciliation',
    'Conservatism',
    'Traditionalism',
    'Patriotism',
    'Nationalism',
    'Imperialism',
    'Colonialism',
    'Expansionism',
    'Isolationism',
    'Pacifism',
    'Militarism',
    'Aggression',
    'Violence',
    'Hatred',
    'Revenge',
    'Retribution',
    'Punishment',
    'Justice',
    'Retaliation',
    'War',
    'Peace',
    'Reconciliation',
    'Tolerance',
    'Acceptance',
    'Diversity',
    'Multiculturalism',
    'Inclusivity',
    'Integration',
    'Segregation',
    'Separatism',
    'Exclusion',
    'Discrimination',
    'Prejudice',
    'Stereotype',
    'Racism',
    'Sexism',
    'Homophobia',
    'Transphobia',
    'Xenophobia',
    'Intolerance',
    'Persecution',
    'Genocide',
    'Holocaust',
    'Slavery',
    'Oppression',
    'Exploitation',
    'Dehumanization',
    'Empowerment',
    'Equality',
    'Equity',
    'Justice',
    'Freedom',
    'Democracy',
    'Republic',
    'Monarchy',
    'Aristocracy',
    'Oligarchy',
    'Dictatorship',
    'Authoritarianism',
    'Totalitarianism',
    'Communism',
    'Socialism',
    'Capitalism',
    'Market',
    'Economy',
    'Industry',
    'Labor',
    'Production',
    'Consumption',
    'Innovation',
    'Technology',
    'Science',
    'Research',
    'Knowledge',
    'Wisdom',
    'Insight',
    'Enlightenment',
    'Perception',
    'Understanding',
    'Clarity',
    'Simplicity',
    'Complexity',
    'Abstraction',
    'Concretion',
    'Realism',
    'Idealism',
    'Materialism',
    'Spirituality',
    'Religion',
    'Atheism',
    'Agnosticism',
    'Humanism',
]

def get_model(name):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if name == 'uae-large':
        model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        tokenizer = AutoTokenizer.from_pretrained('WhereIsAI/UAE-Large-V1')
        dimensions = 1024

    if name == 'bge-large':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        dimensions = 1024

    if name == 'bge-base':
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        dimensions = 768

    if name == 'gte-large':
        model = SentenceTransformer('thenlper/gte-large', device=device)
        tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')
        dimensions = 1024

    if name == 'ember':
        model = SentenceTransformer('llmrails/ember-v1', device=device)
        tokenizer = AutoTokenizer.from_pretrained('llmrails/ember-v1')
        dimensions = 1024

    return model, tokenizer, dimensions


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


# Convert text to embedding
def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return [model.encode(text, show_progress_bar=False)]
    
# Extract embeddings from each row
def content_embeddings(model, df, size, model_name, tokenizer, header=False, randow_row=False):
    
    all_embs = np.empty((0, size), dtype=np.float32)

    for _, row in df.iterrows():
        if (header == False and randow_row == False):
            text = " ".join(map(str, row.values.flatten().tolist()))
        elif(randow_row == True):
            text = " ".join(map(str, row.values.flatten().tolist()))
        elif(header == True):
            text = " ".join(map(str, row.index.tolist()))

        batch_dict = tokenizer(text,  max_length=512, return_attention_mask=False, padding=False, truncation=True)
        # Filter that the row has no more than 512 tokens
        if len(batch_dict['input_ids']) < 512:
            # Create embedding from chunks
            embs = enconde_text(model_name, model, text)
            all_embs = np.append(all_embs, embs, axis=0)
    """
    if (header == False):
        sentences = [" ".join(map(str, row.values.flatten().tolist())) for _, row in df.iterrows()]
    else:
        sentences = [" ".join(map(str, row.index.tolist())) for _, row in df.iterrows()]

    all_embs = model.encode_multi_process(sentences, pool)
    """
    return all_embs


def recover_data(index):
    ids = np.arange(index.ntotal).astype('int64')
    base_embs = []

    for id in ids:
        base_embs.append(index.reconstruct_n(int(id), 1)[0])

    return np.array(base_embs)

