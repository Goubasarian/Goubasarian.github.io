export const SITE = {
  website: "https://Goubasarian.github.io", // replace this with your deployed domain
  author: "Emil Goubasarian",
  profile: "https://Goubasarian.dev/",
  desc: "Hello everyone! My name is Emil, and I am an aspiring AI researcher.",
  title: "Emil's Blog",
  ogImage: "og_image_one.png",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Edit page",
    url: "https://github.com/Goubasarian/Goubasarian.github.io/edit/main/src/data/blog/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Bangkok", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
