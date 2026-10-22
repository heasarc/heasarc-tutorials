window.dataLayer = window.dataLayer || [];
function gtag(){ dataLayer.push(arguments); }
gtag('consent', 'default', {
  analytics_storage: 'denied'
});

document.addEventListener('DOMContentLoaded', function () {
  CookieConsent.run({
    categories: {
      analytics: {
        autoClear: {
          cookies: [
            { name: /^_ga/ },
            { name: '_gid' }
          ]
        }
      }
    },

    guiOptions: {
      consentModal: {
        layout: 'box',
        position: 'bottom right'
      }
    },

    language: {
      default: 'en',
      translations: {
        en: {
          consentModal: {
            title: 'Analytics Consent',
            description: 'HEASARC-Tutorials uses Google Analytics to understand how visitors use it. Do you consent to analytics tracking?',
            acceptAllBtn: 'Accept',
            acceptNecessaryBtn: 'Reject',
            showPreferencesBtn: 'Manage preferences'
          },
          preferencesModal: {
            title: 'Analytics Preferences',
            acceptAllBtn: 'Accept',
            acceptNecessaryBtn: 'Reject',
            savePreferencesBtn: 'Save preferences',
            sections: [
              {
                title: 'Analytics',
                description: 'Used to understand site usage via Google Analytics.',
                linkedCategory: 'analytics'
              }
            ]
          }
        }
      }
    },

    onConsent: ({ cookie }) => updateGAConsent(cookie),
    onChange: ({ cookie }) => updateGAConsent(cookie)
  });
});

function updateGAConsent(cookie) {
  const granted = cookie.categories.includes('analytics');
  gtag('consent', 'update', {
    analytics_storage: granted ? 'granted' : 'denied'
  });
}
